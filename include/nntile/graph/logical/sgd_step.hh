/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/logical/sgd_step.hh
 * Logical graph SGD step operation.
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <string>

// Include other NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

struct SgdStepAttrs
{
    Index num_iter = 0;
    Scalar momentum = 0.0;
    Scalar lr = 0.01;
    Scalar weight_decay = 0.0;
    Scalar dampening = 0.0;
    bool nesterov = false;
};

//! SGD optimizer step: p = sgd_step(grad, velocity, p)
//! @param num_iter Current iteration number
//! @param momentum Momentum factor
//! @param lr Learning rate
//! @param weight_decay Weight decay factor
//! @param dampening Dampening factor
//! @param nesterov Whether to use Nesterov momentum
//! @param grad Gradient tensor
//! @param velocity Velocity tensor (momentum buffer)
//! @param p Parameter tensor (modified in-place)
void sgd_step(
    Index num_iter,
    Scalar momentum,
    Scalar lr,
    Scalar weight_decay,
    Scalar dampening,
    bool nesterov,
    LogicalGraph::TensorNode& grad,
    LogicalGraph::TensorNode& velocity,
    LogicalGraph::TensorNode& p
);

} // namespace nntile::graph
