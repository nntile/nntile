/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/logical/pow_inplace.cc
 * Logical graph pow_inplace operation.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/logical/pow_inplace.hh"

// Include standard headers
#include <stdexcept>
#include <utility>

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! Power in-place: x = alpha * (x ^ exp)
void pow_inplace(
    LogicalGraph::TensorNode& x,
    Scalar alpha,
    Scalar exp)
{
    OpAttrs attrs = PowAttrs{alpha, exp};
    x.graph().add_op(
        OpType::POW_INPLACE,
        attrs,
        {&x},
        {&x}
    );
}

} // namespace nntile::graph