"""Hardware capability checks shared by the Studio API."""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def cuda_available() -> bool:
    """Return whether the installed PyTorch runtime can use a CUDA device."""
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False
