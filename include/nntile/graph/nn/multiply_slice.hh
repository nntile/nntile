/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/nn/multiply_slice.hh
 * NNGraph multiply_slice autograd operation.
 *
 * Forward: output = alpha * src (broadcast along axis)
 * Backward: grad_src += alpha * sum_slice(grad_out)
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <string>

// NNTile headers
#include <nntile/graph/nn/graph_op_node.hh>
#include <nntile/graph/tensor/multiply_slice.hh>

namespace nntile::graph
{

//! MultiplySlice op: output = alpha * src (broadcast). PyTorch-style.
struct NNMultiplySliceOp : NNGraph::OpNode
{
    Scalar alpha;
    Index axis;
    Index axis_size;
    NNGraph::TensorNode* src = nullptr;

    NNMultiplySliceOp(NNGraph::TensorNode* src_,
                      Scalar alpha_,
                      Index axis_,
                      Index axis_size_)
        : alpha(alpha_), axis(axis_), axis_size(axis_size_), src(src_)
    {
        inputs_ = {src};
    }

    NNGraph::TensorNode* forward(const std::string& output_name);
    void backward() const override;
};

NNGraph::TensorNode* multiply_slice(
    Scalar alpha,
    NNGraph::TensorNode* src,
    const std::string& output_name,
    Index axis,
    Index axis_size);

} // namespace nntile::graph
