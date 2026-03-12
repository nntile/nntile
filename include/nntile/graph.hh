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
 * TensorGraph, NNGraph, I/O, Modules, and Optimizers.
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/graph/tensor.hh>
#include <nntile/graph/nn.hh>
#include <nntile/graph/io.hh>
#include <nntile/graph/kv_cache.hh>
#include <nntile/graph/module.hh>
#include <nntile/graph/optim.hh>
