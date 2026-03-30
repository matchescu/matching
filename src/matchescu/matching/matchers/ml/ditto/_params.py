from matchescu.matching.matchers.ml.training import ModelTrainingParams


class DittoModelTrainingParams(ModelTrainingParams):
    frozen_layer_count: int = 0
    alpha_aug: float = 0.8
