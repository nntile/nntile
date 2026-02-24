/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/logical/subtract_indexed_outputs.cc
 * Logical graph subtract_indexed_outputs operation.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/logical/subtract_indexed_outputs.hh"

// Include standard headers
#include <stdexcept>
#include <utility>

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! Subtract indexed outputs operation: subtract val from elements indexed by labels
void subtract_indexed_outputs(
    LogicalGraph::TensorNode& labels,
    LogicalGraph::TensorNode& x,
    Scalar val,
    Index ignore_index)
{
    if(&labels.graph() != &x.graph())
    {
        throw std::invalid_argument(
            "subtract_indexed_outputs: tensors must belong to the same graph");
    }

    if(labels.dtype() != DataType::INT64)
    {
        throw std::invalid_argument(
            "subtract_indexed_outputs: labels tensor must have int64 dtype");
    }

    OpAttrs attrs = SubtractIndexedOutputsAttrs{val, ignore_index};
    labels.graph().add_op(
        OpType::SUBTRACT_INDEXED_OUTPUTS,
        attrs,
        {&labels, &x},
        {&x}
    );
}

} // namespace nntile::graph
