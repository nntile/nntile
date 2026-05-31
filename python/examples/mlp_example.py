#!/usr/bin/env python3
# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# @file python/examples/mlp_example.py
# Port of nntile/examples/graph_mlp_example.cc
#
# @version 1.1.0

"""Trainable MLP forward/backward demo (graph API)."""

from __future__ import annotations

import sys
import time

import numpy as np

import nntile
from nntile import (
    ActivationType,
    Context,
    DataType,
    Mlp,
    NNGraph,
    Runtime,
    TileGraph,
)


def main() -> int:
    _ctx = Context(
        1,
        0,
        0,
        '/tmp/nntile_ooc',
        16777216,
        0,
        'localhost',
        5001,
        0,
    )

    graph = NNGraph('MLP_Graph')
    mlp = Mlp(graph, 'mlp', 8, 16, 4, ActivationType.GELU, DataType.FP32)

    input_tensor = graph.tensor([4, 8], DataType.FP32, True)
    input_tensor.set_name('external_input')
    input_tensor.mark_input(True)

    output_tensor = mlp.forward(input_tensor)
    output_tensor.mark_output(True)

    grad_output = graph.get_or_create_grad(
        output_tensor, 'external_grad_output')
    nntile.nn.fill(1.0, grad_output)

    mlp.fc1().weight_tensor().mark_input(True)
    mlp.fc2().weight_tensor().mark_input(True)

    output_tensor.backward()

    mlp.fc1().weight_tensor().grad.mark_output(True)
    mlp.fc2().weight_tensor().grad.mark_output(True)
    if input_tensor.has_grad:
        input_tensor.grad.mark_output(True)

    print('Graph structure:')
    print(graph)

    tile_graph = TileGraph.from_tensor_graph(graph.tensor_graph())
    runtime = Runtime(tile_graph)
    runtime.compile()

    rng = np.random.default_rng()
    input_data = rng.normal(0.0, 1.0, size=(4 * 8,)).astype(np.float32)
    w1_data = rng.normal(0.0, 0.1, size=(8 * 16,)).astype(np.float32)
    w2_data = rng.normal(0.0, 0.1, size=(16 * 4,)).astype(np.float32)

    runtime.bind_data(input_tensor, input_data)
    runtime.bind_data(mlp.fc1().weight_tensor().data, w1_data)
    runtime.bind_data(mlp.fc2().weight_tensor().data, w2_data)

    print('=== MLP Forward/Backward Pass ===')
    t0 = time.perf_counter()
    runtime.execute()
    runtime.wait()
    us = (time.perf_counter() - t0) * 1e6
    print(f'Graph execution time: {us:.0f} microseconds')

    output_data = runtime.get_output(output_tensor)
    print('Sample output values:', ' '.join(f'{v:.6f}' for v in output_data[:8]),
          '...')

    grad_w1 = runtime.get_output(mlp.fc1().weight_tensor().grad)
    grad_w2 = runtime.get_output(mlp.fc2().weight_tensor().grad)
    print(f'Weight1 grad size: {grad_w1.size}')
    print(f'Weight2 grad size: {grad_w2.size}')
    if input_tensor.has_grad:
        grad_input = runtime.get_output(input_tensor.grad)
        print(f'Input grad size: {grad_input.size}')
    else:
        print('Input grad not available.')

    print('\nMLP module successfully created and graphs built!')
    return 0


if __name__ == '__main__':
    sys.exit(main())
