/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/nn/concat.hh
 * NNGraph concat operation: output = concat(a, b, axis)
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <string>

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/nn/graph_op_node.hh>

namespace nntile::graph
{

//! Concat: output = concat(a, b, axis). Inference-only (no backward).
NNGraph::TensorNode* concat(
    NNGraph::TensorNode* a,
    NNGraph::TensorNode* b,
    Index axis,
    const std::string& output_name);

} // namespace nntile::graph
