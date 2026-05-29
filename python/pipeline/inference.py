"""Shared graph inference pipeline helpers."""

from __future__ import annotations

from typing import Any


def run_inference(model: Any, runtime: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(
        'Port inference loop from wrappers/python after bindings are complete')
