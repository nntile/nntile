/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/logical/hypot_scalar_inverse.cc
 * Logical graph hypot_scalar_inverse operation.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/logical/hypot_scalar_inverse.hh"

// Include standard headers
#include <stdexcept>
#include <utility>

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! Hypot scalar inverse operation: y = 1.0 / hypot(eps, alpha * y)
void hypot_scalar_inverse(
    LogicalGraph::TensorNode& x,
    Scalar eps,
    Scalar alpha)
{
    OpAttrs attrs = HypotScalarInverseAttrs{eps, alpha};
    x.graph().add_op(
        OpType::HYPOT_SCALAR_INVERSE,
        attrs,
        {&x},
        {&x}
    );
}

} // namespace nntile::graph
