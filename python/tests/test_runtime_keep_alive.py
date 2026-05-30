"""GraphRuntime must keep the parent NNGraph alive (pybind keep_alive)."""

from __future__ import annotations

import gc
import weakref

import nntile
from nntile import ActivationType, Context, DataType, Mlp, NNGraph


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

    graph_ref = weakref.ref(graph)
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
