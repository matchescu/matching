from matchescu.matching.matchers.ml.core import ModelTrainingParams


class MultiClassTrainingParams(ModelTrainingParams):
    frozen_layer_count: int = 0
    alpha_aug: float = 0.8
    dropout_p: float = 0.2
    weight_decay: float = 0.01
    output_size: int = 4
