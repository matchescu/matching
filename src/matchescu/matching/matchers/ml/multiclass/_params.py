from matchescu.matching.matchers.ml.core import ModelTrainingParams

from ._types import ArchitectureType, HeadType, LossType


class MultiClassTrainingParams(ModelTrainingParams):
    frozen_layer_count: int = 8
    dropout_p: float = 0.2
    lr_decay_factor: float = 0.95
    weight_decay: float = 0.01
    reverse_penalty_weight: float = 0.0
    head_type: HeadType = HeadType.NONE
    architecture: ArchitectureType = ArchitectureType.BERT
    loss_type: LossType = LossType.WEIGHTED_CE
