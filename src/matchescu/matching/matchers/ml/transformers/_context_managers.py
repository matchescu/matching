from contextlib import contextmanager
import transformers.utils.logging as tl


@contextmanager
def suppress_transformer_modeling_utils_warnings():
    """Temporarily suppress transformers model loading warnings."""
    verbosity = tl.get_verbosity()
    progress_enabled = tl.is_progress_bar_enabled()
    try:
        tl.disable_progress_bar()
        tl.set_verbosity_error()
        yield
    finally:
        tl.set_verbosity(verbosity)
        if progress_enabled:
            tl.enable_progress_bar()
