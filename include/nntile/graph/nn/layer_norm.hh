/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/nn/layer_norm.hh
 * NNGraph LayerNorm autograd operation.
 *
 * Forward: y = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta
 * Backward: grad_x, grad_gamma, grad_beta
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

//! LayerNorm op: y = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta.
struct NNLayerNormOp : NNGraph::OpNode
{
    Index axis;
    Scalar eps;
    int redux;
    NNGraph::TensorNode* x = nullptr;
    NNGraph::TensorNode* gamma = nullptr;
    NNGraph::TensorNode* beta = nullptr;

    NNLayerNormOp() = default;
    NNLayerNormOp(NNGraph::TensorNode* x_,
                  NNGraph::TensorNode* gamma_,
                  NNGraph::TensorNode* beta_,
                  Index axis_ = 0,
                  Scalar eps_ = 1e-5,
                  int redux_ = 0)
        : axis(axis_), eps(eps_), redux(redux_)
        , x(x_), gamma(gamma_), beta(beta_)
    {
        inputs_ = {x, gamma, beta};
    }

    NNGraph::TensorNode* forward(const std::string& output_name);
    void backward() const override;
};

NNGraph::TensorNode* layer_norm(
    NNGraph::TensorNode* x,
    NNGraph::TensorNode* gamma,
    NNGraph::TensorNode* beta,
    const std::string& output_name,
    Index axis = 0,
    Scalar eps = 1e-5,
    int redux = 0);

} // namespace nntile::graph
