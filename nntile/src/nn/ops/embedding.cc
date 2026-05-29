#include <nntile/common.hh>
/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/nn_graph/embedding.cc
 * NNGraph embedding autograd implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/nn/ops/embedding.hh"

#include "nntile/nn/graph_data_node.hh"
#include "nntile/nn/nn_grad_slot_name.hh"
#include "nntile/tensor/ops/clear.hh"
#include "nntile/tensor/ops/embedding.hh"
#include "nntile/tensor/ops/embedding_backward.hh"

#include <stdexcept>

namespace nntile
{

NNGraph::TensorNode *NNEmbeddingOp::forward()
{
    if (index == nullptr || vocab == nullptr)
    {
        throw std::invalid_argument(
            "NNEmbeddingOp::forward: index, vocab must be non-null");
    }
    NNGraph *graph = vocab->graph();
    bool out_requires_grad = any_input_requires_grad({vocab});

    TensorGraph::TensorNode *embed_data =
        tensor_graph::embedding(index->data(), vocab->data(), axis);
    NNGraph::TensorNode *embed = graph->tensor(embed_data, out_requires_grad);
    outputs_ = {embed};
    return embed;
}

void NNEmbeddingOp::backward() const
{
    NNGraph::TensorNode *out = output();
    if (out == nullptr)
    {
        return;
    }
    NNGraph *graph = out->graph();
    NNGraph::TensorNode *grad_out = out->grad();
    if (grad_out == nullptr)
    {
        return;
    }
    if (vocab == nullptr || !vocab->requires_grad())
    {
        return;
    }

    auto [grad_vocab, is_first] =
        graph->get_or_create_grad(vocab, nn_grad_slot_name(vocab));
    if (is_first)
    {
        tensor_graph::clear(grad_vocab->data());
    }
    tensor_graph::embedding_backward(
        index->data(), grad_out->data(), grad_vocab->data(), axis, redux);
}

NNGraph::TensorNode *embedding(NNGraph::TensorNode *index,
    NNGraph::TensorNode *vocab,
    Index axis,
    int redux)
{
    if (index == nullptr || vocab == nullptr)
    {
        throw std::invalid_argument(
            "embedding: index, vocab must be non-null");
    }
    if (index->dtype() != DataType::INT64)
    {
        throw std::invalid_argument("embedding: index must have INT64 dtype");
    }
    NNGraph *graph = vocab->graph();
    auto op = std::make_shared<NNEmbeddingOp>(index, vocab, axis, redux);
    NNGraph::TensorNode *embed = op->forward();
    graph->register_op(std::move(op));
    return embed;
}

} // namespace nntile
