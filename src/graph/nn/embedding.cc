/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/nn/embedding.cc
 * NNGraph embedding autograd implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/nn/embedding.hh"
#include "nntile/graph/nn/graph_data_node.hh"

#include <stdexcept>

#include "nntile/graph/tensor/clear.hh"
#include "nntile/graph/tensor/embedding.hh"
#include "nntile/graph/tensor/embedding_backward.hh"

namespace nntile::graph
{

NNGraph::TensorNode* NNEmbeddingOp::forward(const std::string& output_name)
{
    if(index == nullptr || vocab == nullptr)
    {
        throw std::invalid_argument(
            "NNEmbeddingOp::forward: index, vocab must be non-null");
    }
    NNGraph* graph = vocab->graph();
    bool out_requires_grad = any_input_requires_grad({vocab});

    std::vector<Index> embed_shape = index->shape();
    if(vocab->ndim() != 2)
    {
        throw std::invalid_argument(
            "NNEmbeddingOp::forward: vocab must be 2D");
    }
    embed_shape.push_back(vocab->shape()[1]);

    NNGraph::TensorNode* embed = graph->tensor(
        std::move(embed_shape), output_name, vocab->dtype(), out_requires_grad);
    outputs_ = {embed};

    graph::embedding(index->data(), vocab->data(), embed->data(), axis);
    return embed;
}

void NNEmbeddingOp::backward() const
{
    NNGraph::TensorNode* out = output();
    if(out == nullptr)
    {
        return;
    }
    NNGraph* graph = out->graph();
    NNGraph::TensorNode* grad_out = out->grad();
    if(grad_out == nullptr)
    {
        return;
    }
    if(vocab == nullptr || !vocab->requires_grad())
    {
        return;
    }

    auto [grad_vocab, is_first] =
        graph->get_or_create_grad(vocab, vocab->name() + "_grad");
    if(is_first)
    {
        graph::clear(grad_vocab->data());
    }
    graph::embedding_backward(index->data(), grad_out->data(),
                              grad_vocab->data(), axis, redux);
}

NNGraph::TensorNode* embedding(
    NNGraph::TensorNode* index,
    NNGraph::TensorNode* vocab,
    const std::string& output_name,
    Index axis,
    int redux)
{
    if(index == nullptr || vocab == nullptr)
    {
        throw std::invalid_argument("embedding: index, vocab must be non-null");
    }
    if(index->dtype() != DataType::INT64)
    {
        throw std::invalid_argument("embedding: index must have INT64 dtype");
    }
    NNGraph* graph = vocab->graph();
    auto op = std::make_shared<NNEmbeddingOp>(index, vocab, axis, redux);
    NNGraph::TensorNode* embed = op->forward(output_name);
    register_op(*graph, std::move(op));
    return embed;
}

} // namespace nntile::graph
