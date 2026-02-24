/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/logical/adamw_step.cc
 * Logical graph AdamW step operation.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/logical/adamw_step.hh"

// Include standard headers
#include <stdexcept>
#include <utility>

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! AdamW optimizer step: p = adamw_step(grad, first_moment, second_moment, p)
void adamw_step(
    Index num_iter,
    Scalar beta_1,
    Scalar beta_2,
    Scalar eps,
    Scalar lr,
    Scalar weight_decay,
    LogicalGraph::TensorNode& grad,
    LogicalGraph::TensorNode& first_moment,
    LogicalGraph::TensorNode& second_moment,
    LogicalGraph::TensorNode& p)
{
    if(&grad.graph() != &first_moment.graph() || &grad.graph() != &second_moment.graph() || &grad.graph() != &p.graph())
    {
        throw std::invalid_argument(
            "adamw_step: tensors must belong to the same graph");
    }

    if(grad.dtype() != first_moment.dtype() || grad.dtype() != second_moment.dtype() || grad.dtype() != p.dtype())
    {
        throw std::invalid_argument(
            "adamw_step: all tensors must have the same dtype");
    }

    if(grad.shape() != first_moment.shape() || grad.shape() != second_moment.shape() || grad.shape() != p.shape())
    {
        throw std::invalid_argument(
            "adamw_step: all tensors must have the same shape");
    }

    OpAttrs attrs = AdamStepAttrs{num_iter, beta_1, beta_2, eps, lr, weight_decay};
    grad.graph().add_op(
        OpType::ADAMW_STEP,
        attrs,
        {&grad, &first_moment, &second_moment, &p},
        {&first_moment, &second_moment, &p}
    );
}

} // namespace nntile::graph
