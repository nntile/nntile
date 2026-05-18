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

//! Concat op: output = concat(a, b, axis). Forward uses TensorGraph CONCAT;
//! backward is not implemented yet (throws when gradient must flow).
struct NNConcatOp : NNGraph::OpNode
{
    NNGraph::TensorNode* a = nullptr;
    NNGraph::TensorNode* b = nullptr;
    Index axis = 0;

    NNConcatOp(NNGraph::TensorNode* a_, NNGraph::TensorNode* b_, Index axis_)
        : a(a_), b(b_), axis(axis_)
    {
        inputs_ = {a, b};
    }

    NNGraph::TensorNode* forward(const std::string& output_name);
    void backward() const override;
};

//! Concat: output = concat(a, b, axis). Inference forward; autograd registration
//! follows other NN ops, but backward through concat is not supported yet.
NNGraph::TensorNode* concat(
    NNGraph::TensorNode* a,
    NNGraph::TensorNode* b,
    Index axis,
    const std::string& output_name);

} // namespace nntile::graph
