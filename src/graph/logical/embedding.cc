/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/logical/embedding.cc
 * Logical graph embedding operation.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/logical/embedding.hh"

// Include standard headers
#include <stdexcept>
#include <utility>

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! Embedding lookup: y = embedding(x, vocab)
LogicalGraph::TensorNode& embedding(
    LogicalGraph::TensorNode& index,
    LogicalGraph::TensorNode& vocab,
    const std::string& output_name,
    Index axis)
{
    if(&index.graph() != &vocab.graph())
    {
        throw std::invalid_argument(
            "embedding: tensors must belong to the same graph");
    }

    if(index.dtype() != DataType::INT64)
    {
        throw std::invalid_argument(
            "embedding: index tensor must have int64 dtype");
    }

    if(axis < 0 || axis >= index.ndim())
    {
        throw std::invalid_argument(
            "embedding: axis out of bounds");
    }

    // Compute output shape
    std::vector<Index> output_shape = index.shape();
    output_shape[axis] = vocab.shape()[axis];

    // Create output tensor
    LogicalGraph::TensorNode& output = index.graph().tensor(
        std::move(output_shape),
        output_name,
        vocab.dtype());

    // Create operation attributes
    OpAttrs attrs = EmbeddingAttrs{axis};

    // Add operation to graph
    index.graph().add_op(
        OpType::EMBEDDING,
        attrs,
        {&index, &vocab},
        {&output}
    );

    return output;
}

} // namespace nntile::graph