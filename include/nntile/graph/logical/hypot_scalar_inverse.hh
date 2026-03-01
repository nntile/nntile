/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/logical/hypot_scalar_inverse.hh
 * Logical graph hypot_scalar_inverse operation.
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

struct HypotScalarInverseAttrs
{
    Scalar eps = 0.0;
    Scalar alpha = 1.0;
};

//! Hypot scalar inverse operation: y = 1.0 / hypot(eps, alpha * y)
//! @param x Input/output tensor (modified in-place)
//! @param eps Epsilon value for numerical stability
//! @param alpha Scaling factor
void hypot_scalar_inverse(
    LogicalGraph::TensorNode& x,
    Scalar eps = 0.0,
    Scalar alpha = 1.0
);

} // namespace nntile::graph
