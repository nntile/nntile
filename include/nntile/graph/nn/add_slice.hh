/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/nn/add_slice.hh
 * NNGraph add_slice autograd operation.
 *
 * Forward: output = alpha * src1 + beta * src2 (src1 broadcast along axis)
 * Backward: grad_src1 += alpha * sum_slice(grad_out), grad_src2 += beta * grad_out
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <string>

// NNTile headers
#include <nntile/graph/nn/graph_op_node.hh>
#include <nntile/graph/tensor/add_slice.hh>

namespace nntile::graph
{

//! AddSlice op: output = alpha*src1 + beta*src2 (src1 slice broadcast). PyTorch-style.
struct NNAddSliceOp : NNGraph::OpNode
{
    Scalar alpha;
    Scalar beta;
    Index axis;
    NNGraph::TensorNode* src1 = nullptr;
    NNGraph::TensorNode* src2 = nullptr;

    NNAddSliceOp(NNGraph::TensorNode* src1_,
                 NNGraph::TensorNode* src2_,
                 Scalar alpha_, Scalar beta_,
                 Index axis_)
        : alpha(alpha_), beta(beta_), axis(axis_)
        , src1(src1_), src2(src2_)
    {
        inputs_ = {src1, src2};
    }

    NNGraph::TensorNode* forward(const std::string& output_name);
    void backward() const override;
};

NNGraph::TensorNode* add_slice(
    Scalar alpha,
    NNGraph::TensorNode* src1,
    Scalar beta,
    NNGraph::TensorNode* src2,
    const std::string& output_name,
    Index axis);

} // namespace nntile::graph
