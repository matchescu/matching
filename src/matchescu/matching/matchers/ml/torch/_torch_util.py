import logging
import os
import random

import numpy as np
import torch

CUBLAS_WORKSPACE_VAR = "CUBLAS_WORKSPACE_CONFIG"
CUBLAS_DETERMINISTIC_WORKSPACE = ":4096:8"
HASH_SEED_VAR = "PYTHONHASHSEED"
EXACT_FP32 = "ieee"

_log = logging.getLogger(__name__)


def _configure_cuda_determinism() -> None:
    os.environ.setdefault(CUBLAS_WORKSPACE_VAR, CUBLAS_DETERMINISTIC_WORKSPACE)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.conv.fp32_precision = EXACT_FP32
    torch.backends.cuda.matmul.fp32_precision = EXACT_FP32


def _warn_if_hash_seed_unset() -> None:
    if HASH_SEED_VAR not in os.environ:
        _log.warning(
            "%s is not set: set iteration order may differ between runs. "
            "Export it before starting the interpreter.",
            HASH_SEED_VAR,
        )


def set_random_seed(seed: int) -> None:
    """Seed the random number generators and request deterministic execution.

    Seeding alone does not make training repeatable on GPUs: some kernels
    pick algorithms by timing or accumulate with atomics. This function also
    switches PyTorch to the platform primitives that remove those sources of
    variation, so a fixed seed gives the same result on the same hardware and
    software stack.

    .. note::
        ``torch.manual_seed`` seeds the CPU generator and every accelerator
        generator, so no per-device seeding call is needed. Deterministic
        algorithms run in warn-only mode: an operation that has no
        deterministic implementation logs a warning instead of aborting a
        long training run. ``CUBLAS_WORKSPACE_CONFIG`` is only read when
        cuBLAS initialises, so this function must run before the first CUDA
        operation. An existing value of that variable is kept.

    .. warning::
        ``PYTHONHASHSEED`` is read when the interpreter starts. It cannot be
        fixed from here, so a warning is logged if it is missing.

    :param seed: Seed for ``random``, ``numpy`` and ``torch``.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    _configure_cuda_determinism()
    torch.use_deterministic_algorithms(True, warn_only=True)
    _warn_if_hash_seed_unset()
