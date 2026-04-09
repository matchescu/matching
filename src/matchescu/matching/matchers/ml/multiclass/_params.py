from matchescu.matching.matchers.ml.core import ModelTrainingParams


class MultiClassTrainingParams(ModelTrainingParams):
    frozen_layer_count: int = 8
    neg_pos_ratio: float = 8.0
    match_bridge_ratio: float = 2.0
    alpha_aug: float = 0.8
    dropout_p: float = 0.2
    weight_decay: float = 0.01
    output_size: int = 4
    decay_factor: float = 0.95
