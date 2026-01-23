/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/logical/embedding_backward.cc
 * Logical graph embedding backward operation.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/logical/embedding_backward.hh"

// Include standard headers
#include <stdexcept>
#include <utility>

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! Embedding backward: vocab += embedding_backward(embed, index, vocab)
void embedding_backward(
    LogicalGraph::TensorNode& embed,
    LogicalGraph::TensorNode& index,
    LogicalGraph::TensorNode& vocab,
    Index axis)
{
    if(&embed.graph() != &index.graph() || &embed.graph() != &vocab.graph())
    {
        throw std::invalid_argument(
            "embedding_backward: tensors must belong to the same graph");
    }

    if(index.dtype() != DataType::INT64)
    {
        throw std::invalid_argument(
            "embedding_backward: index tensor must have int64 dtype");
    }

    if(embed.dtype() != vocab.dtype())
    {
        throw std::invalid_argument(
            "embedding_backward: embed and vocab must have the same dtype");
    }

    if(axis < 0 || axis >= embed.ndim())
    {
        throw std::invalid_argument(
            "embedding_backward: axis out of bounds");
    }

    OpAttrs attrs = EmbeddingAttrs{axis};
    embed.graph().add_op(
        OpType::EMBEDDING_BACKWARD,
        attrs,
        {&embed, &index, &vocab},
        {&vocab}
    );
}

} // namespace nntile::graph