/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/logical/adamw_step.hh
 * Logical graph AdamW step operation.
 *
 * @version 1.1.0
 * */

#pragma once

// Include other NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

struct AdamWAttrs
{
    Index num_iter = 0;
    Scalar beta_1 = 0.9;
    Scalar beta_2 = 0.999;
    Scalar eps = 1e-8;
    Scalar lr = 0.001;
    Scalar weight_decay = 0.0;
};

//! AdamW optimizer step
//! @param num_iter Current iteration number
//! @param beta_1 First moment decay rate
//! @param beta_2 Second moment decay rate
//! @param eps Small constant for numerical stability
//! @param lr Learning rate
//! @param weight_decay Weight decay factor
//! @param grad Gradient tensor
//! @param first_moment First moment tensor
//! @param second_moment Second moment tensor
//! @param p Parameter tensor (modified in-place)
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
    LogicalGraph::TensorNode& p
);

} // namespace nntile::graph
