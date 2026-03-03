/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/nn/graph.hh
 * NNGraph - graph with gradients. Purely symbolic; use TensorGraph::Runtime for
 * execution.
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/graph/nn/graph_decl.hh>
#include <nntile/graph/nn/graph_data_node.hh>
#include <nntile/graph/nn/graph_op_node.hh>
