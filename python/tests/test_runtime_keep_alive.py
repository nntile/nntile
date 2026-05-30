"""GraphRuntime must keep the parent NNGraph alive (pybind keep_alive)."""

from __future__ import annotations

import gc
import weakref

import pytest

from nntile import ActivationType, DataType, Mlp, NNGraph


def _require_nngraph_weakref(graph: NNGraph):
    """NNGraph must support weak refs (pybind11_object / keep_alive testing)."""
    try:
        return weakref.ref(graph)
    except TypeError as exc:
        raise AssertionError(
            'NNGraph binding does not support weak references; '
            'cannot verify runtime() keep_alive',
        ) from exc


def test_runtime_view_keeps_nngraph_alive(nntile_context) -> None:
    del nntile_context
    graph = NNGraph('keep_alive_test')
    mlp = Mlp(graph, 'mlp', 4, 8, 2, ActivationType.GELU, DataType.FP32)
    x = graph.tensor([2, 4], DataType.FP32, True)
    x.mark_input(True)
    y = mlp.forward(x)
    y.mark_output(True)
    graph.finish_phase()
    graph.lower_and_compile()

    graph_ref = _require_nngraph_weakref(graph)
    runtime = graph.runtime()
    del graph
    gc.collect()

    assert graph_ref() is not None, (
        'NNGraph was collected while GraphRuntime still exists'
    )
    assert runtime.is_compiled
    del runtime
    gc.collect()
    assert graph_ref() is None
