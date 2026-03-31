from matchescu.matching.matchers.ml.core import ModelTrainingParams


class DeepMatcherModelTrainingParams(ModelTrainingParams):
    vocab_size: int
    embedding_dim: int
    num_attributes: int = 3
    hidden_size: int = 100
    dropout: float = 0.2
