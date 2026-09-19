from enum import Enum


class HeadType(str, Enum):
    NONE = "none"
    ABS = "abs"
    ASYMMETRIC = "asymmetric"


class ArchitectureType(str, Enum):
    BERT = "bert"
    BERT_CROSS_ATTN = "bert_cross_attn"
    BERT_PER_ATTR_CROSS_ATTN = "bert_per_attr_cross_attn"


class LossType(str, Enum):
    WEIGHTED_CE = "weighted_ce"
    FOCAL = "focal"
