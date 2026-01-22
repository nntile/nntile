/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph.hh
 * Convenience header for NNTile graph system.
 *
 * @version 1.1.0
 * */

#pragma once

// Include related NNTile headers - core classes
#include <nntile/graph/logical_graph.hh>
#include <nntile/graph/nn_graph.hh>
#include <nntile/graph/compiled_graph.hh>

// Include logical graph operations
#include <nntile/graph/logical/gemm.hh>
#include <nntile/graph/logical/gelu.hh>
#include <nntile/graph/logical/gelu_backward.hh>

// Include compiled graph operations
#include <nntile/graph/compiled/gemm.hh>
#include <nntile/graph/compiled/gelu.hh>
#include <nntile/graph/compiled/gelu_backward.hh>
