/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/nn/rms_norm.hh
 * NNGraph RMSNorm autograd operation.
 *
 * Forward: y = gamma * (x / sqrt(mean(x^2) + eps))
 * Backward: grad_x, grad_gamma
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <string>

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/nn/graph_op_node.hh>

namespace nntile::graph
{

//! RMSNorm op: y = gamma * (x / sqrt(mean(x^2) + eps)).
//! PyTorch-style: outputs created in forward().
struct NNRmsNormOp : NNGraph::OpNode
{
    Index axis;
    Scalar eps;
    int redux;
    NNGraph::TensorNode* x = nullptr;
    NNGraph::TensorNode* gamma = nullptr;

    NNRmsNormOp() = default;
    NNRmsNormOp(NNGraph::TensorNode* x_,
                NNGraph::TensorNode* gamma_,
                Index axis_ = 0,
                Scalar eps_ = 1e-6,
                int redux_ = 0)
        : axis(axis_), eps(eps_), redux(redux_), x(x_), gamma(gamma_)
    {
        inputs_ = {x, gamma};
    }

    NNGraph::TensorNode* forward(const std::string& output_name);
    void backward() const override;
};

NNGraph::TensorNode* rms_norm(
    NNGraph::TensorNode* x,
    NNGraph::TensorNode* gamma,
    const std::string& output_name,
    Index axis = 0,
    Scalar eps = 1e-6,
    int redux = 0);

} // namespace nntile::graph
