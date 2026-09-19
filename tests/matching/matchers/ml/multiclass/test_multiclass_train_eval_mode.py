import pytest

from matchescu.matching.matchers.ml.multiclass._module import MultiClassModule
from matchescu.matching.matchers.ml.multiclass._types import (
    ArchitectureType,
)


@pytest.fixture
def module(make_params):
    return MultiClassModule(
        make_params(
            head_type="asymmetric",
            architecture=ArchitectureType.BERT_PER_ATTR_CROSS_ATTN,
        )
    )


def test_eval_sets_cross_attention_to_eval_mode(module):
    module.train(True)
    module.eval()

    assert module.cross_attention.training is False


def test_eval_sets_classifier_to_eval_mode(module):
    module.train(True)
    module.eval()

    assert module.classifier.training is False


def test_train_sets_module_training_flag(module):
    module.eval()
    module.train(True)

    assert module.training is True


def test_eval_sets_module_training_flag(module):
    module.train(True)
    module.eval()

    assert module.training is False
