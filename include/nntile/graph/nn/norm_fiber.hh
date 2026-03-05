/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/nn/norm_fiber.hh
 * NNGraph norm_fiber autograd operation.
 *
 * Forward: output = alpha * norm_fiber(x)
 * Backward: not implemented (throws)
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <string>

// NNTile headers
#include <nntile/graph/nn/graph_op_node.hh>
#include <nntile/graph/tensor/norm_fiber.hh>

namespace nntile::graph
{

//! NormFiber op: output = alpha * norm_fiber(x). PyTorch-style. Always fresh output.
struct NNNormFiberOp : NNGraph::OpNode
{
    Scalar alpha;
    Index axis;
    Index batch_ndim;
    int redux;
    NNGraph::TensorNode* x = nullptr;

    NNNormFiberOp(NNGraph::TensorNode* x_,
                  Index axis_, Index batch_ndim_, int redux_,
                  Scalar alpha_)
        : alpha(alpha_), axis(axis_), batch_ndim(batch_ndim_)
        , redux(redux_), x(x_)
    {
        inputs_ = {x};
    }

    NNGraph::TensorNode* forward(const std::string& output_name);
    void backward() const override;
};

NNGraph::TensorNode* norm_fiber(
    NNGraph::TensorNode* x,
    const std::string& output_name,
    Index axis,
    Index batch_ndim,
    int redux,
    Scalar alpha = 1.0);

} // namespace nntile::graph
