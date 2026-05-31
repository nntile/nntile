# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file python/tests/test_mlp_example.py
#
# @version 1.1.0

"""Run the MLP example logic (mirrors graph_mlp_example.cc)."""

from __future__ import annotations

import numpy as np

import nntile
from nntile import (
    ActivationType,
    DataType,
    Mlp,
    NNGraph,
    Runtime,
    TileGraph,
)


def test_mlp_forward_backward(nntile_context):
    del nntile_context
    graph = NNGraph('MLP_Graph_test')
    mlp = Mlp(graph, 'mlp', 4, 8, 2, ActivationType.GELU, DataType.FP32)

    x = graph.tensor([2, 4], DataType.FP32, True)
    x.mark_input(True)
    y = mlp.forward(x)
    y.mark_output(True)

    grad_y = graph.get_or_create_grad(y, 'grad_y')
    nntile.nn.fill(1.0, grad_y)
    mlp.fc1().weight_tensor().mark_input(True)
    mlp.fc2().weight_tensor().mark_input(True)
    y.backward()
    mlp.fc1().weight_tensor().grad.mark_output(True)

    tile_graph = TileGraph.from_tensor_graph(graph.tensor_graph())
    runtime = Runtime(tile_graph)
    runtime.compile_with_round_robin_schedule()

    rng = np.random.default_rng(0)
    runtime.bind_data(x, rng.normal(size=8).astype(np.float32))
    runtime.bind_data(
        mlp.fc1().weight_tensor().data,
        rng.normal(size=32).astype(np.float32),
    )
    runtime.bind_data(
        mlp.fc2().weight_tensor().data,
        rng.normal(size=16).astype(np.float32),
    )
    runtime.execute()
    runtime.wait()

    out = runtime.get_output(y)
    assert out.size == 4
    gw = runtime.get_output(mlp.fc1().weight_tensor().grad)
    assert gw.size == 32
