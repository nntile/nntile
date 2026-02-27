/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/nn_graph_ops.hh
 * NNGraph operations - free function overloads for NNGraph::TensorNode.
 *
 * Mirrors the LogicalGraph API structure: individual ops in nn_graph/,
 * this file includes them all.
 *
 * @version 1.1.0
 * */

#pragma once

// Include NNGraph operation overloads (one file per op, like logical/)
#include <nntile/graph/nn_graph/add.hh>
#include <nntile/graph/nn_graph/add_fiber.hh>
#include <nntile/graph/nn_graph/gemm.hh>
#include <nntile/graph/nn_graph/gelu.hh>
#include <nntile/graph/nn_graph/gelu_backward.hh>
#include <nntile/graph/nn_graph/sum_fiber.hh>  // autograd SumFiber
