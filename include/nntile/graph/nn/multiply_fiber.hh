/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/nn/multiply_fiber.hh
 * NNGraph multiply_fiber autograd operation.
 *
 * Forward: output = alpha * src1 * src2 (element-wise along fiber)
 * Backward: grad_src1 += alpha * grad_out * src2, grad_src2 += alpha * grad_out * src1
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <string>

// NNTile headers
#include <nntile/graph/nn/graph_op_node.hh>
#include <nntile/graph/tensor/multiply_fiber.hh>

namespace nntile::graph
{

//! MultiplyFiber op: output = alpha * src1 * src2. PyTorch-style.
struct NNMultiplyFiberOp : NNGraph::OpNode
{
    Scalar alpha;
    Index axis;
    NNGraph::TensorNode* src1 = nullptr;
    NNGraph::TensorNode* src2 = nullptr;

    NNMultiplyFiberOp(NNGraph::TensorNode* src1_,
                     NNGraph::TensorNode* src2_,
                     Scalar alpha_,
                     Index axis_)
        : alpha(alpha_), axis(axis_), src1(src1_), src2(src2_)
    {
        inputs_ = {src1, src2};
    }

    NNGraph::TensorNode* forward(const std::string& output_name);
    void backward() const override;
};

NNGraph::TensorNode* multiply_fiber(
    Scalar alpha,
    NNGraph::TensorNode* src1,
    NNGraph::TensorNode* src2,
    const std::string& output_name,
    Index axis);

} // namespace nntile::graph
