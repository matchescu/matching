from matchescu.matching.matchers.ml.core import ModelTrainingParams


class DeepERParams(ModelTrainingParams):
    num_attributes: int = 3
    lstm_hidden_size: int = 150
    similarity_hidden_size: int = 50
    frozen_layer_count: int = 8
    output_size: int = 2
