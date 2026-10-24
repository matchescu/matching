import logging


def _build_null_logger() -> logging.Logger:
    logger = logging.getLogger("matchescu.matching.null")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return logger


NULL_LOGGER = _build_null_logger()


def resolve_logger(logger: logging.Logger | None) -> logging.Logger:
    return logger if logger is not None else NULL_LOGGER
