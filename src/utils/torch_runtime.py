from __future__ import annotations

import random

import numpy as np
import torch


_MIN_TORCH_SEED = -(2**63)
_MAX_TORCH_SEED = 2**64 - 1


def seed_everything(seed: int | None, *, rank: int = 0) -> int | None:

    if seed is None:
        return None
    effective = int(seed) + int(rank)
    if not _MIN_TORCH_SEED <= effective <= _MAX_TORCH_SEED:
        raise ValueError(
            "effective seed (seed + rank) must be in "
            f"[{_MIN_TORCH_SEED}, {_MAX_TORCH_SEED}], got {effective}"
        )
    random.seed(effective)
    np.random.seed(effective % (2**32))
    torch.manual_seed(effective)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective)
    return effective


def configure_torch_runtime(device: torch.device) -> None:

    if device.type != "cpu":
        return
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
