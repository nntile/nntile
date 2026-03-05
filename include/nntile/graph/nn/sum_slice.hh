/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/nn/sum_slice.hh
 * NNGraph sum_slice autograd operation.
 *
 * Forward: output = alpha * sum_slice(x) (fresh output, no in-place)
 * Backward: grad_x += alpha * add_slice_inplace(grad_out) (broadcast)
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <string>

// NNTile headers
#include <nntile/graph/nn/graph_op_node.hh>
#include <nntile/graph/tensor/sum_slice.hh>

namespace nntile::graph
{

//! SumSlice op: output = alpha * sum_slice(x). PyTorch-style. Always fresh output.
struct NNSumSliceOp : NNGraph::OpNode
{
    Scalar alpha;
    Index axis;
    int redux;
    NNGraph::TensorNode* x = nullptr;

    NNSumSliceOp(NNGraph::TensorNode* x_,
                 Index axis_, int redux_,
                 Scalar alpha_)
        : alpha(alpha_), axis(axis_), redux(redux_), x(x_)
    {
        inputs_ = {x};
    }

    NNGraph::TensorNode* forward(const std::string& output_name);
    void backward() const override;
};

NNGraph::TensorNode* sum_slice(
    NNGraph::TensorNode* x,
    const std::string& output_name,
    Index axis,
    int redux,
    Scalar alpha);

} // namespace nntile::graph
