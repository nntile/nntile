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
//! loss = scale * norm(x)^2, gradient = 2*scale*x
//! scale=1.0 gives total loss, scale=1/num_values gives mean loss
class MseLoss : public Module
{
private:
    graph::NNGraph::TensorNode* input_tensor_ = nullptr;
    graph::NNGraph::TensorNode* loss_tensor_ = nullptr;
    graph::DataType dtype_;
    Scalar scale_;

public:
    //! Constructor
    MseLoss(graph::NNGraph* graph,
            const std::string& name,
            graph::DataType dtype = graph::DataType::FP32,
            Scalar scale = 1.0);

    //! Build forward: loss = scale * norm(x)^2 (scalar)
    graph::NNGraph::TensorNode& forward(
        graph::NNGraph::TensorNode& input);

    //! Forward: calls forward
    graph::NNGraph::TensorNode* operator()(graph::NNGraph::TensorNode* input)
    {
        return forward(input);
    }

    std::string repr() const override;

    //! Get scale parameter
    Scalar scale() const { return scale_; }
};

} // namespace nntile::module
