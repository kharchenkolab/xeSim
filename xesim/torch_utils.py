from __future__ import annotations

import logging
from typing import Literal

import torch

DeviceName = Literal["auto", "cpu", "cuda", "mps"]


def get_device(requested: str | None = None, logger: logging.Logger | None = None) -> torch.device:
    """Resolve a PyTorch device with CUDA/MPS preference and safe CPU fallback."""

    log = logger or logging.getLogger(__name__)
    choice = "auto" if requested is None else requested.lower()
    if choice not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported device '{requested}'. Choose auto, cpu, cuda, or mps.")

    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        log.warning("CUDA was requested but is unavailable; falling back to CPU.")
        return torch.device("cpu")
    if choice == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_built():
            log.warning("MPS was requested and PyTorch was built with MPS, but MPS is unavailable.")
        else:
            log.warning("MPS was requested but this PyTorch build does not include MPS.")
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps"):
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.backends.mps.is_built():
            log.info("PyTorch was built with MPS, but MPS is unavailable; using CPU.")
    return torch.device("cpu")
