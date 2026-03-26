import logging
import sys


def new_training_logger():
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(name)s(%(thread)d)[%(levelname)s] %(asctime)s » %(message)s",
    )
    return logging.root.getChild("ditto-training")


log = new_training_logger()
