/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/nn_graph/ops/norm.hh
 * NNGraph norm autograd operation.
 *
 * Forward: output = alpha * norm(x) (scalar)
 * Backward: not implemented (throws)
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <string>

// NNTile headers
#include <nntile/nn_graph/graph_op_node.hh>
#include <nntile/tensor_graph/ops/norm.hh>

namespace nntile
{

//! Norm op: output = alpha * norm(x) (scalar). PyTorch-style. Always fresh
//! output.
struct NNNormOp : NNGraph::OpNode
{
    Scalar alpha;
    NNGraph::TensorNode *x = nullptr;

    NNNormOp(NNGraph::TensorNode *x_, Scalar alpha_) : alpha(alpha_), x(x_)
    {
        inputs_ = {x};
    }

    NNGraph::TensorNode *forward();
    void backward() const override;
};

NNGraph::TensorNode *norm(NNGraph::TensorNode *x, Scalar alpha = 1.0);

} // namespace nntile
