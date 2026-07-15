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
#include "nntile/tensor/ops/embedding.hh"
#include "nntile/tensor/ops/embedding_backward.hh"

#include <stdexcept>

namespace nntile
{

namespace
{

Index normalize_embedding_axis(Index axis, Index index_ndim)
{
    Index norm_axis = (axis < 0) ? index_ndim : axis;
    if (norm_axis < 0 || norm_axis > index_ndim)
    {
        throw std::invalid_argument(
            "embedding: axis out of range for index ndim");
    }
    return norm_axis;
}

std::vector<Index> embedding_output_shape(
    const std::vector<Index> &index_shape,
    Index embed_dim,
    Index axis)
{
    std::vector<Index> out;
    out.reserve(index_shape.size() + 1);
    for (Index i = 0; i < axis; ++i)
    {
        out.push_back(index_shape[static_cast<size_t>(i)]);
    }
    out.push_back(embed_dim);
    for (Index i = axis; i < static_cast<Index>(index_shape.size()); ++i)
    {
        out.push_back(index_shape[static_cast<size_t>(i)]);
    }
    return out;
}

} // anonymous namespace

NNGraph::TensorNode *NNEmbeddingOp::forward()
{
    if (index == nullptr || vocab == nullptr)
    {
        throw std::invalid_argument(
            "NNEmbeddingOp::forward: index, vocab must be non-null");
    }
    NNGraph *graph = vocab->graph();
    bool out_requires_grad = any_input_requires_grad({vocab});

    const Index embed_dim = vocab->shape().back();
    const std::vector<Index> out_shape = embedding_output_shape(
        index->shape(), embed_dim, axis);

    TensorGraph *tensor_graph = vocab->data()->graph();
    TensorGraph::TensorNode *embed_data = tensor_graph->emplace_data(
        out_shape, vocab->data()->dtype());
    tensor::embedding(
        index->data(), vocab->data(), embed_data, axis);
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
    const Scalar beta = is_first ? Scalar{0.0} : Scalar{1.0};
    tensor::embedding_backward(
        index->data(), grad_out->data(), grad_vocab->data(), axis,
        Scalar{1.0}, beta, redux);
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
    const Index norm_axis = normalize_embedding_axis(axis, index->ndim());
    auto op = std::make_shared<NNEmbeddingOp>(index, vocab, norm_axis, redux);
    NNGraph::TensorNode *embed = op->forward();
    graph->register_op(std::move(op));
    return embed;
}

} // namespace nntile
