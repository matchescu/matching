from contextlib import contextmanager
import logging


@contextmanager
def suppress_transformer_modeling_utils_warnings():
    """Temporarily suppress transformers model loading warnings."""
    hf_logger = logging.getLogger("transformers.modeling_utils")
    original_level = hf_logger.level
    hf_logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        hf_logger.setLevel(original_level)
