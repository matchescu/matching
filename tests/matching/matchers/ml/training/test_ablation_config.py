import json
from pathlib import Path

import pytest

from matchescu.matching.matchers.ml.training import TrainingConfig

_DISCOVERY = [
    "matchescu.matching.matchers.ml.multiclass.training",
]


@pytest.fixture
def ablation_config_path(data_dir) -> Path:
    return data_dir / "ablation_config.json"


@pytest.fixture
def ablation_config(ablation_config_path, data_dir) -> TrainingConfig:
    return TrainingConfig.load_json(
        ablation_config_path,
        data_dir=data_dir,
        discovery_packages=_DISCOVERY,
    )


def test_ablation_config_loads(ablation_config):
    assert ablation_config.model_names == ["bert-base-uncased"]
    assert ablation_config.included_datasets == ["amazon-google"]


def test_ablation_config_resolves_base_params(ablation_config):
    params = ablation_config.get(model="bert-base-uncased", dataset="amazon-google")
    assert params.head_type.value == "none"
    assert params.architecture.value == "bert"
    assert params.loss_type.value == "weighted_ce"
    assert params.dir_margin_weight == 0.0
    assert params.frozen_layer_count == 8


def test_ablation_config_defaults_preserve_current_behavior(data_dir):
    """Keep the signed head with optional loss terms disabled by default."""
    raw = {
        "kind": "multiclass",
        "learningRate": 2e-5,
        "batchSize": 32,
        "epochs": 15,
        "models": ["bert-base-uncased"],
        "datasets": {
            "amazon-google": {
                "type": "magellan",
                "directory": "amazon_google_exp_data",
                "leftTraits": "amazon-google",
            }
        },
        "modelConfig": {
            "bert-base-uncased": {
                "modelName": "bert-base-uncased",
                "frozenLayerCount": 8,
            }
        },
    }
    path = data_dir / "ablation_defaults_test.json"
    path.write_text(json.dumps(raw))
    try:
        config = TrainingConfig.load_json(
            path, data_dir=data_dir, discovery_packages=_DISCOVERY
        )
        params = config.get(model="bert-base-uncased", dataset="amazon-google")
        assert params.head_type.value == "signed"
        assert params.architecture.value == "bert"
        assert params.loss_type.value == "focal"
        assert params.focal_gamma == 0.0
        assert params.dir_margin_weight == 0.0
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.parametrize(
    "overrides,expected_head,expected_arch,expected_loss,expected_weight",
    [
        (
            {
                "headType": "none",
                "architecture": "bert",
                "lossType": "weighted_ce",
                "dirMarginWeight": 0.0,
            },
            "none",
            "bert",
            "weighted_ce",
            0.0,
        ),
        (
            {
                "headType": "abs",
                "architecture": "bert",
                "lossType": "weighted_ce",
                "dirMarginWeight": 0.0,
            },
            "abs",
            "bert",
            "weighted_ce",
            0.0,
        ),
        (
            {
                "headType": "signed",
                "architecture": "bert",
                "lossType": "weighted_ce",
                "dirMarginWeight": 0.0,
            },
            "signed",
            "bert",
            "weighted_ce",
            0.0,
        ),
        (
            {
                "headType": "none",
                "architecture": "bert_cross_attn",
                "lossType": "weighted_ce",
                "dirMarginWeight": 0.0,
            },
            "none",
            "bert_cross_attn",
            "weighted_ce",
            0.0,
        ),
        (
            {
                "headType": "none",
                "architecture": "bert_per_attr_cross_attn",
                "lossType": "weighted_ce",
                "dirMarginWeight": 0.0,
            },
            "none",
            "bert_per_attr_cross_attn",
            "weighted_ce",
            0.0,
        ),
        (
            {
                "headType": "none",
                "architecture": "bert",
                "lossType": "focal",
                "dirMarginWeight": 0.0,
            },
            "none",
            "bert",
            "focal",
            0.0,
        ),
        (
            {
                "headType": "none",
                "architecture": "bert",
                "lossType": "weighted_ce",
                "dirMarginWeight": 2.0,
            },
            "none",
            "bert",
            "weighted_ce",
            2.0,
        ),
        (
            {
                "headType": "none",
                "architecture": "bert",
                "lossType": "focal",
                "dirMarginWeight": 2.0,
            },
            "none",
            "bert",
            "focal",
            2.0,
        ),
    ],
)
def test_ablation_param_overrides(
    data_dir,
    overrides,
    expected_head,
    expected_arch,
    expected_loss,
    expected_weight,
):
    """Each ablation scenario's overrides should resolve to the correct params."""
    raw = {
        "kind": "multiclass",
        "learningRate": 2e-5,
        "batchSize": 32,
        "epochs": 15,
        "models": ["bert-base-uncased"],
        "datasets": {
            "amazon-google": {
                "type": "magellan",
                "directory": "amazon_google_exp_data",
                "leftTraits": "amazon-google",
            }
        },
        "modelConfig": {
            "bert-base-uncased": {
                "modelName": "bert-base-uncased",
                "frozenLayerCount": 8,
                **overrides,
            }
        },
    }
    path = data_dir / f"ablation_override_test_{id(overrides)}.json"
    path.write_text(json.dumps(raw))
    try:
        config = TrainingConfig.load_json(
            path, data_dir=data_dir, discovery_packages=_DISCOVERY
        )
        params = config.get(model="bert-base-uncased", dataset="amazon-google")
        assert params.head_type.value == expected_head
        assert params.architecture.value == expected_arch
        assert params.loss_type.value == expected_loss
        assert params.dir_margin_weight == expected_weight
    finally:
        path.unlink(missing_ok=True)
