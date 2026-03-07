/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/adamw_step.hh
 * TensorGraph adamw_step: fused AdamW optimizer step
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/tensor/graph.hh>

namespace nntile::graph::tensor
{

//! AdamW step: Adam with decoupled weight decay
struct TensorAdamwStepOp : TensorGraph::OpNode
{
    Index num_iter;
    Scalar beta_1;
    Scalar beta_2;
    Scalar eps;
    Scalar lr;
    Scalar weight_decay;
    TensorGraph::TensorNode* grad = nullptr;
    TensorGraph::TensorNode* first_moment = nullptr;
    TensorGraph::TensorNode* second_moment = nullptr;
    TensorGraph::TensorNode* p = nullptr;

    TensorAdamwStepOp() = default;
    TensorAdamwStepOp(Index num_iter_, Scalar beta_1_, Scalar beta_2_,
                      Scalar eps_, Scalar lr_, Scalar weight_decay_,
                      TensorGraph::TensorNode* grad_,
                      TensorGraph::TensorNode* first_moment_,
                      TensorGraph::TensorNode* second_moment_,
                      TensorGraph::TensorNode* p_)
        : num_iter(num_iter_), beta_1(beta_1_), beta_2(beta_2_),
          eps(eps_), lr(lr_), weight_decay(weight_decay_),
          grad(grad_), first_moment(first_moment_),
          second_moment(second_moment_), p(p_)
    {
        inputs_ = {grad, first_moment, second_moment, p};
        outputs_ = {first_moment, second_moment, p};
    }

    std::string op_name() const override { return "ADAMW_STEP"; }

    void execute(TensorGraph::Runtime& runtime) const override;

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorAdamwStepOp>(*this);
    }
};

//! AdamW step
void adamw_step(Index num_iter, Scalar beta_1, Scalar beta_2,
                Scalar eps, Scalar lr, Scalar weight_decay,
                TensorGraph::TensorNode* grad,
                TensorGraph::TensorNode* first_moment,
                TensorGraph::TensorNode* second_moment,
                TensorGraph::TensorNode* p);

} // namespace nntile::graph::tensor
