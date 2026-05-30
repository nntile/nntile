/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph.hh
 * Convenience header for the entire NNTile graph API:
 * TensorGraph, TileGraph, NNGraph, I/O, Modules, and Optimizers.
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/common.hh>

// Graph stack: tensor -> tile -> nn (tensor layer has no tile graph dependency).
#include <nntile/tensor.hh>
#include <nntile/tile.hh>
#include <nntile/nn.hh>
#include <nntile/io.hh>
#include <nntile/kv_cache.hh>
#include <nntile/module.hh>
#include <nntile/optim.hh>
