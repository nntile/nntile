/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/nn/ops/hypot.hh
 * NNGraph hypot autograd operation.
 *
 * Forward: output = hypot(alpha*x, beta*y) element-wise
 * Backward: not implemented (throws)
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <string>

// NNTile headers
#include <nntile/nn/graph_op_node.hh>
#include <nntile/tensor/ops/hypot.hh>

namespace nntile
{

//! Hypot op: output = hypot(alpha*x, beta*y). PyTorch-style.
struct NNHypotOp : NNGraph::OpNode
{
    Scalar alpha;
    Scalar beta;
    NNGraph::TensorNode *x = nullptr;
    NNGraph::TensorNode *y = nullptr;

    NNHypotOp(NNGraph::TensorNode *x_,
        NNGraph::TensorNode *y_,
        Scalar alpha_ = 1.0,
        Scalar beta_ = 1.0) :
        alpha(alpha_), beta(beta_), x(x_), y(y_)
    {
        inputs_ = {x, y};
    }

    NNGraph::TensorNode *forward();
    void backward() const override;
};

NNGraph::TensorNode *hypot(NNGraph::TensorNode *x,
    NNGraph::TensorNode *y,
    Scalar alpha = 1.0,
    Scalar beta = 1.0);

} // namespace nntile
