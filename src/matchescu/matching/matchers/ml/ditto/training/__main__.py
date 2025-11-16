import time
import warnings
from contextlib import contextmanager
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Iterable

import click
import humanize
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerFast, DebertaV2TokenizerFast

from matchescu.blocking import TfIdfBlocker
from matchescu.comparison_filtering import is_cross_source_comparison
from matchescu.csg import BinaryComparisonSpaceGenerator, BinaryComparisonSpaceEvaluator
from matchescu.data import Record
from matchescu.data_sources import CsvDataSource
from matchescu.extraction import (
    Traits,
    RecordExtraction,
    single_record,
)
from matchescu.matching.evaluation.datasets import MagellanDataset, MagellanTraits
from matchescu.matching.matchers.ml.ditto._ditto_module import DittoModel
from matchescu.reference_store.id_table import InMemoryIdTable
from matchescu.typing import (
    EntityReferenceIdentifier as RefId,
    EntityReferenceIdFactory,
)

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


def create_comparison_space(id_table, ground_truth, initial_size):
    csg = (
        BinaryComparisonSpaceGenerator()
        .add_blocker(TfIdfBlocker(id_table, 0.23))
        .add_filter(is_cross_source_comparison)
    )
    comparison_space = csg()
    eval_cs = BinaryComparisonSpaceEvaluator(ground_truth, initial_size)
    metrics = eval_cs(comparison_space)
    print(metrics)
    return comparison_space


def _id(records: list[Record], source: str):
    return RefId(records[0][0], source)


def load_data_source(id_table: InMemoryIdTable, data_source: CsvDataSource) -> None:
    extract_references = RecordExtraction(
        data_source, partial(_id, source=data_source.name), single_record
    )
    for ref in extract_references():
        id_table.put(ref)


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
    left_id_factory: EntityReferenceIdFactory,
    right_traits: Traits | None = None,
    right_id_factory: EntityReferenceIdFactory | None = None,
) -> MagellanDataset:
    ds = MagellanDataset(ds_path)
    ds.load_left(left_traits, left_id_factory)
    ds.load_right(right_traits or left_traits, right_id_factory or left_id_factory)
    ds.load_splits()
    return ds


@timer(start_message="serialize+tokenize")
def get_magellan_data_loaders(
    magellan_ds: MagellanDataset,
    tokenizer: PreTrainedTokenizerFast,
    batch_size: int = 32,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds, xv_ds, test_ds = [
        DittoDataset(
            magellan_ds.id_table, split.comparison_space, split.ground_truth, tokenizer
        )
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
def train_on_magellan_data(
    model_save_dir: Path,
    model_name: str,
    tokenizer: PreTrainedTokenizerFast,
    train_params: ModelTrainingParams,
    dataset_dir: Path,
    dataset_name: str,
    traits: Traits,
    id_factory: EntityReferenceIdFactory,
    pair_traits: Traits | None = None,
    pair_id_factory: EntityReferenceIdFactory | None = None,
):
    pair_traits = pair_traits or traits
    pair_id_factory = pair_id_factory or id_factory
    magellan_ds = load_magellan_dataset(
        dataset_dir / dataset_name,
        traits,
        id_factory,
        pair_traits,
        pair_id_factory,
    )
    train, xv, test = get_magellan_data_loaders(
        magellan_ds, tokenizer, train_params.batch_size
    )
    ditto = DittoModel(model_name)
    dataset_logger = log.getChild(dataset_name)
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


def table_a_id(rows: Iterable[Record]) -> RefId:
    return RefId(next(iter(rows))["id"], "tableA")


def table_b_id(rows: Iterable[Record]) -> RefId:
    return RefId(next(iter(rows))["id"], "tableB")


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
        for dataset_name in config.dataset_names:
            ds_model_dir = root_model_dir / dataset_name
            for model_name in config.model_names:
                tokenizer = _new_fast_tokenizer(model_name)
                train_params = config.get(model=model_name, dataset=dataset_name)
                ds_traits = benchmark_dataset_traits[dataset_name]

                train_on_magellan_data(
                    ds_model_dir,
                    model_name,
                    tokenizer,
                    train_params,
                    root_data_dir / "magellan",
                    dataset_name,
                    traits=ds_traits,
                    id_factory=table_a_id,
                    pair_id_factory=table_b_id,
                )


def _new_fast_tokenizer(model_name: str) -> PreTrainedTokenizerFast:
    tokenizer_factory = _MODEL_TOKENIZERS.get(
        model_name, partial(AutoTokenizer.from_pretrained, use_fast=True)
    )
    return tokenizer_factory(model_name)


if __name__ == "__main__":
    run_training()
