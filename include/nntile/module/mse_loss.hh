/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/module/mse_loss.hh
 * MSE loss module: squared Frobenius norm of input.
 *
 * Forward: loss = norm(x)^2 = ||x||_F^2
 * Backward: grad_x = 2*x (no need to set grad on scalar loss)
 *
 * @version 1.1.0
 * */

#pragma once

// Include NNTile headers
#include <nntile/graph.hh>
#include <nntile/module/module.hh>

namespace nntile::module
{

//! MSE loss: squared Frobenius norm of input tensor.
//! loss = norm(x)^2, gradient = 2*x
class MseLoss : public ModuleBase
{
private:
    graph::NNGraph::TensorNode* input_tensor_ = nullptr;
    graph::NNGraph::TensorNode* loss_tensor_ = nullptr;
    graph::DataType dtype_;

public:
    //! Constructor
    MseLoss(graph::NNGraph& graph,
            const std::string& name,
            graph::DataType dtype = graph::DataType::FP32);

    //! Build forward: loss = norm(x)^2 (scalar)
    graph::NNGraph::TensorNode& build_forward(
        graph::NNGraph::TensorNode& input);

    //! Forward: calls build_forward
    graph::NNGraph::TensorNode& operator()(graph::NNGraph::TensorNode& input)
    {
        return build_forward(input);
    }

    //! Build backward: grad_x = 2*x (scalar loss grad is 1.0)
    void build_backward();

    std::string repr() const override;
};

} // namespace nntile::module
