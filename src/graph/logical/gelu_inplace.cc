/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/logical/gelu_inplace.cc
 * Logical graph GeLU in-place operation.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/logical/gelu_inplace.hh"

// Include standard headers
#include <stdexcept>
#include <utility>

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! GeLU in-place: x = gelu(x)
void gelu_inplace(LogicalGraph::TensorNode& x)
{
    OpAttrs attrs = GeluAttrs{};
    x.graph().add_op(
        OpType::GELU_INPLACE,
        attrs,
        {&x},
        {&x}
    );
}

} // namespace nntile::graph