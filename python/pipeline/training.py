"""Shared graph training pipeline helpers (port from wrappers/python)."""

from __future__ import annotations

from typing import Any


def run_training_loop(model: Any, runtime: Any, **kwargs: Any) -> None:
    """Placeholder: bind model forward, lower, execute, optimizer step."""
    raise NotImplementedError(
        'Port training loop from wrappers/python after bindings are complete')
