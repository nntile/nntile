/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/logical/sgd_step.cc
 * Logical graph SGD step operation.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/logical/sgd_step.hh"

// Include standard headers
#include <stdexcept>
#include <utility>

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! SGD optimizer step: p = sgd_step(grad, velocity, p)
void sgd_step(
    Index num_iter,
    Scalar momentum,
    Scalar lr,
    Scalar weight_decay,
    Scalar dampening,
    bool nesterov,
    LogicalGraph::TensorNode& grad,
    LogicalGraph::TensorNode& velocity,
    LogicalGraph::TensorNode& p)
{
    if(&grad.graph() != &velocity.graph() || &grad.graph() != &p.graph())
    {
        throw std::invalid_argument(
            "sgd_step: tensors must belong to the same graph");
    }

    if(grad.dtype() != velocity.dtype() || grad.dtype() != p.dtype())
    {
        throw std::invalid_argument(
            "sgd_step: all tensors must have the same dtype");
    }

    if(grad.shape() != velocity.shape() || grad.shape() != p.shape())
    {
        throw std::invalid_argument(
            "sgd_step: all tensors must have the same shape");
    }

    OpAttrs attrs = SgdStepAttrs{num_iter, momentum, lr, weight_decay, dampening, nesterov};
    grad.graph().add_op(
        OpType::SGD_STEP,
        attrs,
        {&grad, &velocity, &p},
        {&velocity, &p}
    );
}

} // namespace nntile::graph
