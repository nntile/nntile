/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/logical/fill.cc
 * Logical graph fill operation.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/logical/fill.hh"

// Include standard headers
#include <stdexcept>
#include <utility>

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! Fill operation: x = val
void fill(
    Scalar val,
    LogicalGraph::TensorNode& x)
{
    auto attrs = std::make_shared<FillAttrs>(FillAttrs{val});
    x.graph().add_op(
        OpType::FILL,
        attrs,
        {},
        {&x}
    );
}

} // namespace nntile::graph
