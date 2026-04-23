/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/concat.hh
 * TensorGraph concat operation: output = concat(a, b, axis)
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <string>
#include <vector>

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/tensor/graph.hh>

namespace nntile::graph::tensor
{

//! Concat operation: output = concat(a, b, axis)
struct TensorConcatOp : TensorGraph::OpNode
{
    TensorGraph::TensorNode* a = nullptr;
    TensorGraph::TensorNode* b = nullptr;
    TensorGraph::TensorNode* output = nullptr;
    Index axis = 0;

    TensorConcatOp() = default;
    TensorConcatOp(TensorGraph::TensorNode* a_,
                   TensorGraph::TensorNode* b_,
                   TensorGraph::TensorNode* output_,
                   Index axis_)
        : a(a_), b(b_), output(output_), axis(axis_)
    {
        inputs_ = {a, b};
        outputs_ = {output};
    }

    std::string op_name() const override { return "CONCAT"; }

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorConcatOp>(*this);
    }
};

//! Concat: output = concat(a, b, axis). Creates output tensor.
TensorGraph::TensorNode* concat(
    TensorGraph::TensorNode* a,
    TensorGraph::TensorNode* b,
    Index axis,
    const std::string& output_name);

} // namespace nntile::graph::tensor
