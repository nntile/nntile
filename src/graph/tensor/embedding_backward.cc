/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/embedding_backward.cc
 * TensorGraph embedding_backward operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/embedding_backward.hh"

#include <stdexcept>

#include "nntile/graph/tensor.hh"
#include "nntile/tensor/embedding_backward.hh"

namespace nntile::graph::tensor
{



void embedding_backward(TensorGraph::TensorNode* index,
                        TensorGraph::TensorNode* embed,
                        TensorGraph::TensorNode* vocab,
                        Index axis,
                        int redux)
{
    if(index == nullptr || embed == nullptr || vocab == nullptr)
        throw std::invalid_argument("embedding_backward: tensors must be non-null");
    if(index->graph() != embed->graph() || embed->graph() != vocab->graph())
        throw std::invalid_argument("embedding_backward: tensors must belong to same graph");
    if(index->dtype() != DataType::INT64)
        throw std::invalid_argument("embedding_backward: index must have INT64 dtype");
    if(embed->dtype() != vocab->dtype())
        throw std::invalid_argument("embedding_backward: embed and vocab must have same dtype");
    validate_embedding_shape_and_merge(embed, index, vocab,
                                      "embedding_backward");

    auto op = std::make_shared<TensorEmbeddingBackwardOp>(
        index, embed, vocab, axis, redux);
    vocab->graph()->add_op(op);
}

} // namespace nntile::graph::tensor
