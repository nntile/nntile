/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/nn_graph/ops/silu.hh
 * NNGraph SiLU autograd operation.
 *
 * Forward: y = silu(x)
 * Backward: grad_x += silu_backward(x, grad_y)
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <string>

// NNTile headers
#include <nntile/nn_graph/graph_op_node.hh>
#include <nntile/tensor_graph/ops/silu.hh>

namespace nntile
{

//! SiLU op: y = silu(x). PyTorch-style: outputs created in forward().
struct NNSiluOp : NNGraph::OpNode
{
    NNGraph::TensorNode *x = nullptr;

    NNSiluOp() = default;
    explicit NNSiluOp(NNGraph::TensorNode *x_) : x(x_) { inputs_ = {x}; }

    NNGraph::TensorNode *forward();
    void backward() const override;
};

NNGraph::TensorNode *silu(NNGraph::TensorNode *x);

} // namespace nntile
