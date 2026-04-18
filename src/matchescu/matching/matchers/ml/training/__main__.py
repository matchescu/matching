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

from matchescu.matching.evaluation.data.benchmark._base import BenchmarkData
from matchescu.matching.matchers.ml.core import ModelTrainingParams
from matchescu.matching.matchers.ml.deeper import DeepERModule
from matchescu.matching.matchers.ml.deeper.training import DeepERTrainer, DeepERDataset
from matchescu.matching.matchers.ml.deepmatcher import DeepMatcherModule
from matchescu.matching.matchers.ml.deepmatcher.training import (
    DeepMatcherDataset,
    DeepMatcherTrainer,
)
from matchescu.matching.matchers.ml.ditto import DittoModel
from matchescu.matching.matchers.ml.ditto.training import DittoDataset, DittoTrainer
from matchescu.matching.matchers.ml.multiclass import MultiClassModule
from matchescu.matching.matchers.ml.multiclass.training import (
    MultiClassTrainer,
    AsymmetricMultiClassDataset,
)
from matchescu.matching.matchers.ml.training import BaseTrainer, BaseEvaluator

from matchescu.matching.matchers.ml.training._config import (
    TrainingConfig,
    DEFAULT_DATA_DIR,
    DEFAULT_MODEL_DIR,
    MATCHERS_ML_PACKAGE,
)
from matchescu.matching.matchers.ml.training._dataset import TDataset
from matchescu.matching.matchers.ml.training._logging import log

_MODEL_TOKENIZERS = {
    "microsoft/deberta-v3-base": DebertaV2TokenizerFast.from_pretrained,
}
_TRAINER_MAPPINGS: dict[type, tuple[type, type]] = {
    DeepMatcherTrainer: (DeepMatcherModule, DeepMatcherDataset),
    DittoTrainer: (DittoModel, DittoDataset),
    DeepERTrainer: (DeepERModule, DeepERDataset),
    MultiClassTrainer: (MultiClassModule, AsymmetricMultiClassDataset),
}


@contextmanager
def timer(start_message: str):
    log.info(start_message)
    time_start = time.time()
    yield
    time_end = time.time()
    duration = humanize.naturaldelta(timedelta(seconds=(time_end - time_start)))
    log.info("%s time elapsed: %s", start_message, duration)


@timer(start_message="serialize+tokenize")
def get_benchmark_data_loaders(
    ds_cls: type[TDataset],
    benchmark_data: BenchmarkData,
    tokenizer: PreTrainedTokenizerFast,
    train_params: ModelTrainingParams,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds, xv_ds, test_ds = [
        ds_cls(benchmark_data.id_table, split, tokenizer)
        for split_name, split in benchmark_data.splits.items()
    ]
    sampler = (
        train_ds.get_weighted_sampler()
        if isinstance(train_ds, AsymmetricMultiClassDataset)
        else None
    )
    shuffle_training_data = sampler is None
    return (
        train_ds.get_data_loader(
            train_params.batch_size, shuffle_training_data, sampler
        ),
        xv_ds.get_data_loader(train_params.batch_size * 16),
        test_ds.get_data_loader(train_params.batch_size * 16),
    )


@timer(start_message="train ditto")
def train_on_benchmark_data[TParams](
    model_save_dir: Path,
    model_name: str,
    trainer_cls: type[BaseTrainer],
    evaluator_cls: type[BaseEvaluator],
    benchmark_data: BenchmarkData,
    tokenizer: PreTrainedTokenizerFast,
    train_params: TParams,
):
    if (model_and_ds := _TRAINER_MAPPINGS.get(trainer_cls)) is None:
        raise RuntimeError(f"unsupported trainer: {trainer_cls.__qualname__}")
    model_cls, ds_cls = model_and_ds
    train, xv, test = get_benchmark_data_loaders(
        ds_cls, benchmark_data, tokenizer, train_params
    )
    matcher_model = model_cls(train_params)
    dataset_logger = log.getChild(benchmark_data.name)
    trainer = trainer_cls(
        model_name,
        train_params,
        model_save_dir,
        logger=dataset_logger,
    )
    tb_log_dir = model_save_dir / model_name / "tensorboard"
    with evaluator_cls(model_name, xv, test, tb_log_dir, dataset_logger) as evaluator:
        trainer.run_training(matcher_model, train, evaluator, True)


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
    config = TrainingConfig.load_json(
        config_path,
        data_dir=root_data_dir,
        discovery_packages=[
            f"{MATCHERS_ML_PACKAGE}.ditto.training",
            f"{MATCHERS_ML_PACKAGE}.deepmatcher.training",
            f"{MATCHERS_ML_PACKAGE}.deeper.training",
            f"{MATCHERS_ML_PACKAGE}.multiclass.training",
        ],
    )
    with warnings.catch_warnings(action="ignore"):
        for dataset_name in config.included_datasets:
            ds_model_dir = root_model_dir / dataset_name
            data_builder = config.data_builders[dataset_name]
            benchmark_data = data_builder.load_data().load_splits().create()

            for model_name in config.model_names:
                train_params = config.get(model=model_name, dataset=dataset_name)
                tokenizer = _new_fast_tokenizer(train_params.model_name or model_name)
                trainer_cls = config.get_trainer(model_name)
                eval_cls = config.get_evaluator(model_name)

                train_on_benchmark_data(
                    ds_model_dir,
                    model_name,
                    trainer_cls,
                    eval_cls,
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
