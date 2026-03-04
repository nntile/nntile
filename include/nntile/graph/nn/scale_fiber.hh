/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/nn/scale_fiber.hh
 * NNGraph scale_fiber autograd operation.
 *
 * Forward: output = alpha * src (broadcast along fiber)
 * Backward: grad_src += alpha * sum_fiber(grad_out)
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <string>
#include <vector>

// NNTile headers
#include <nntile/graph/nn/graph_op_node.hh>
#include <nntile/graph/tensor/scale_fiber.hh>

namespace nntile::graph
{

//! ScaleFiber op: output = alpha * src (broadcast). PyTorch-style.
struct NNScaleFiberOp : NNGraph::OpNode
{
    Scalar alpha;
    Index axis;
    Index batch_ndim;
    std::vector<Index> dst_shape;
    NNGraph::TensorNode* src = nullptr;

    NNScaleFiberOp(NNGraph::TensorNode* src_,
                   Scalar alpha_,
                   Index axis_,
                   Index batch_ndim_,
                   std::vector<Index> dst_shape_)
        : alpha(alpha_), axis(axis_), batch_ndim(batch_ndim_)
        , dst_shape(std::move(dst_shape_)), src(src_)
    {
        inputs_ = {src};
    }

    NNGraph::TensorNode* forward(const std::string& output_name);
    void backward() const override;
};

NNGraph::TensorNode* scale_fiber(
    Scalar alpha,
    NNGraph::TensorNode* src,
    const std::string& output_name,
    const std::vector<Index>& dst_shape,
    Index axis,
    Index batch_ndim);

} // namespace nntile::graph
