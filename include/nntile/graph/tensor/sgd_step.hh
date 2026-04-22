/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/sgd_step.hh
 * TensorGraph sgd_step: fused SGD with momentum step
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/tensor/graph.hh>

namespace nntile::graph
{
struct LoweringContext;
}

namespace nntile::graph::tensor
{

//! SGD step: p = p - lr*grad (with momentum, weight decay, etc.)
struct TensorSgdStepOp : TensorGraph::OpNode
{
    mutable Index num_iter;
    Scalar momentum;
    Scalar lr;
    Scalar weight_decay;
    Scalar dampening;
    bool nesterov = false;
    TensorGraph::TensorNode* grad = nullptr;
    TensorGraph::TensorNode* velocity = nullptr;
    TensorGraph::TensorNode* p = nullptr;

    TensorSgdStepOp() = default;
    TensorSgdStepOp(Index num_iter_, Scalar momentum_, Scalar lr_,
                   Scalar weight_decay_, Scalar dampening_, bool nesterov_,
                   TensorGraph::TensorNode* grad_,
                   TensorGraph::TensorNode* velocity_,
                   TensorGraph::TensorNode* p_)
        : num_iter(num_iter_), momentum(momentum_), lr(lr_),
          weight_decay(weight_decay_), dampening(dampening_), nesterov(nesterov_),
          grad(grad_), velocity(velocity_), p(p_)
    {
        inputs_ = {grad, velocity, p};
        outputs_ = {velocity, p};
    }

    std::string op_name() const override { return "SGD_STEP"; }

    void execute(TensorGraph::Runtime& runtime) const override;

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorSgdStepOp>(*this);
    }
    void lower_to_tile(const LoweringContext& ctx) const override;

};

//! SGD step
void sgd_step(Index num_iter, Scalar momentum, Scalar lr,
              Scalar weight_decay, Scalar dampening, bool nesterov,
              TensorGraph::TensorNode* grad,
              TensorGraph::TensorNode* velocity,
              TensorGraph::TensorNode* p);

} // namespace nntile::graph::tensor
