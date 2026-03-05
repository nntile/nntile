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
 * Forward: output = alpha * slice * tensor (slice broadcast along axis)
 * Backward: grad_slice += alpha * sum_slice(grad_out * tensor), grad_tensor += alpha * grad_out * slice
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

//! MultiplySlice op: output = alpha * slice * tensor (slice broadcast). PyTorch-style.
struct NNMultiplySliceOp : NNGraph::OpNode
{
    Scalar alpha;
    Index axis;
    NNGraph::TensorNode* slice = nullptr;
    NNGraph::TensorNode* tensor = nullptr;

    NNMultiplySliceOp(NNGraph::TensorNode* slice_,
                     NNGraph::TensorNode* tensor_,
                     Scalar alpha_,
                     Index axis_)
        : alpha(alpha_), axis(axis_), slice(slice_), tensor(tensor_)
    {
        inputs_ = {slice, tensor};
    }

    NNGraph::TensorNode* forward(const std::string& output_name);
    void backward() const override;
};

NNGraph::TensorNode* multiply_slice(
    Scalar alpha,
    NNGraph::TensorNode* slice,
    NNGraph::TensorNode* tensor,
    const std::string& output_name,
    Index axis);

} // namespace nntile::graph
