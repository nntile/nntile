/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/logical/log_scalar.cc
 * Logical graph log_scalar operation.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/logical/log_scalar.hh"

// Include standard headers
#include <stdexcept>
#include <utility>
#include <string>

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! Log scalar operation: log value with given name
void log_scalar(
    LogicalGraph::TensorNode& x,
    const std::string& name)
{
    OpAttrs attrs = LogScalarAttrs{name};
    x.graph().add_op(
        OpType::LOG_SCALAR,
        attrs,
        {&x},
        {}
    );
}

} // namespace nntile::graph