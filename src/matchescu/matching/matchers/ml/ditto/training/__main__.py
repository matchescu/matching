import time
import warnings
from contextlib import contextmanager
from datetime import timedelta
from functools import partial
from pathlib import Path

import click
import humanize
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerFast, DebertaV2TokenizerFast

from matchescu.extraction import Traits
from matchescu.matching.evaluation.data import MagellanBenchmarkData, MagellanTraits
from matchescu.matching.matchers.ml.ditto._ditto_module import DittoModel
from matchescu.matching.matchers.ml.ditto.training._config import (
    TrainingConfig,
    ModelTrainingParams,
    DEFAULT_MODEL_DIR,
    DEFAULT_DATA_DIR,
)
from matchescu.matching.matchers.ml.ditto.training._datasets import DittoDataset
from matchescu.matching.matchers.ml.ditto.training._trainer import DittoTrainer
from matchescu.matching.matchers.ml.ditto.training._evaluator import TrainingEvaluator
from matchescu.matching.matchers.ml.ditto.training._logging import log


_MODEL_TOKENIZERS = {
    "microsoft/deberta-v3-base": DebertaV2TokenizerFast.from_pretrained,
}


@contextmanager
def timer(start_message: str):
    log.info(start_message)
    time_start = time.time()
    yield
    time_end = time.time()
    duration = humanize.naturaldelta(timedelta(seconds=(time_end - time_start)))
    log.info("%s time elapsed: %s", start_message, duration)


@timer(start_message="load dataset")
def load_magellan_dataset(
    ds_path: Path,
    left_traits: Traits,
    right_traits: Traits | None = None,
) -> MagellanBenchmarkData:
    ds = MagellanBenchmarkData(ds_path)
    ds.load_left(left_traits)
    ds.load_right(right_traits or left_traits)
    ds.load_splits()
    return ds


@timer(start_message="serialize+tokenize")
def get_magellan_data_loaders(
    magellan_ds: MagellanBenchmarkData,
    tokenizer: PreTrainedTokenizerFast,
    batch_size: int = 32,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds, xv_ds, test_ds = [
        DittoDataset(magellan_ds.id_table, split, tokenizer)
        for split in [
            magellan_ds.train_split,
            magellan_ds.valid_split,
            magellan_ds.test_split,
        ]
    ]
    return (
        train_ds.get_data_loader(batch_size, shuffle=True),
        train_ds.get_data_loader(batch_size * 16),
        train_ds.get_data_loader(batch_size * 16),
    )


@timer(start_message="train ditto")
def train_on_benchmark_data(
    model_save_dir: Path,
    model_name: str,
    benchmark_data: MagellanBenchmarkData,
    tokenizer: PreTrainedTokenizerFast,
    train_params: ModelTrainingParams,
):
    train, xv, test = get_magellan_data_loaders(
        benchmark_data, tokenizer, train_params.batch_size
    )
    ditto = DittoModel(model_name)
    dataset_logger = log.getChild(benchmark_data.name)
    trainer = DittoTrainer(
        model_name,
        model_save_dir,
        learning_rate=train_params.learning_rate,
        epochs=train_params.epochs,
        logger=dataset_logger,
    )
    tb_log_dir = model_save_dir / model_name / "tensorboard"
    with TrainingEvaluator(
        model_name, xv, test, tb_log_dir, dataset_logger
    ) as evaluator:
        trainer.run_training(ditto, train, evaluator, True)


@click.command
@click.option(
    "-M",
    "--model-dir",
    "root_model_dir",
    required=True,
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    default=DEFAULT_MODEL_DIR,
)
@click.option(
    "-D",
    "--dataset-dir",
    "root_data_dir",
    required=True,
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    default=DEFAULT_DATA_DIR,
)
@click.option(
    "-f",
    "--config-file",
    "config_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    default=DEFAULT_MODEL_DIR / "config.json",
)
def run_training(
    root_model_dir: Path,
    root_data_dir: Path,
    config_path: Path,
) -> None:
    root_model_dir = Path(root_model_dir)
    root_data_dir = Path(root_data_dir)
    benchmark_dataset_traits = MagellanTraits()
    config = TrainingConfig.load_json(config_path)
    with warnings.catch_warnings(action="ignore"):
        for dataset_name in config.included_datasets:
            ds_path = root_data_dir / "magellan" / dataset_name
            ds_model_dir = root_model_dir / dataset_name
            for model_name in config.model_names:
                tokenizer = _new_fast_tokenizer(model_name)
                train_params = config.get(model=model_name, dataset=dataset_name)
                ds_traits = benchmark_dataset_traits[dataset_name]
                benchmark_data = load_magellan_dataset(ds_path, ds_traits)

                train_on_benchmark_data(
                    ds_model_dir,
                    model_name,
                    benchmark_data,
                    tokenizer,
                    train_params,
                )


def _new_fast_tokenizer(model_name: str) -> PreTrainedTokenizerFast:
    tokenizer_factory = _MODEL_TOKENIZERS.get(
        model_name, partial(AutoTokenizer.from_pretrained, use_fast=True)
    )
    return tokenizer_factory(model_name)


if __name__ == "__main__":
    run_training()
