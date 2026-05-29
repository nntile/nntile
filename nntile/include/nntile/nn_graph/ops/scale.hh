/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/nn_graph/ops/scale.hh
 * NNGraph scale autograd operation.
 *
 * Forward: output = alpha * src
 * Backward: grad_src += alpha * grad_out
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <string>

// NNTile headers
#include <nntile/nn_graph/graph_op_node.hh>
#include <nntile/tensor_graph/ops/scale.hh>

namespace nntile
{

//! Scale op: output = alpha*src. PyTorch-style.
struct NNScaleOp : NNGraph::OpNode
{
    Scalar alpha;
    NNGraph::TensorNode *src = nullptr;

    NNScaleOp(NNGraph::TensorNode *src_, Scalar alpha_) :
        alpha(alpha_), src(src_)
    {
        inputs_ = {src};
    }

    NNGraph::TensorNode *forward();
    void backward() const override;
};

NNGraph::TensorNode *scale(Scalar alpha, NNGraph::TensorNode *src);

} // namespace nntile
