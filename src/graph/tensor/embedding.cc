/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/embedding.cc
 * TensorGraph embedding operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/embedding.hh"

#include <stdexcept>

#include "nntile/graph/tensor.hh"
#include "nntile/tensor/embedding.hh"

namespace nntile::graph::tensor
{

namespace
{

template<typename T>
void run_embedding(TensorGraph::Runtime& runtime,
                  TensorGraph::TensorNode* index,
                  TensorGraph::TensorNode* vocab,
                  TensorGraph::TensorNode* embed,
                  Index axis)
{
    auto& index_t = runtime.get_tensor<nntile::int64_t>(index);
    auto& vocab_t = runtime.get_tensor<T>(vocab);
    auto& embed_t = runtime.get_tensor<T>(embed);
    nntile::tensor::embedding<T>(index_t, vocab_t, embed_t, axis);
}

} // namespace

TensorGraph::TensorNode* embedding(TensorGraph::TensorNode* index,
                                    TensorGraph::TensorNode* vocab,
                                    const std::string& output_name,
                                    Index axis)
{
    if(index == nullptr || vocab == nullptr)
        throw std::invalid_argument("embedding: tensors must be non-null");
    if(index->graph() != vocab->graph())
        throw std::invalid_argument("embedding: tensors must belong to same graph");
    if(index->dtype() != DataType::INT64)
        throw std::invalid_argument("embedding: index must have INT64 dtype");
    // Output shape: index.shape + (vocab.shape[0],) at axis
    // NNTile layout: vocab [embed_dim, num_embeddings]; embed.shape[axis] == vocab.shape[0]
    std::vector<Index> embed_shape = index->shape();
    if(vocab->ndim() != 2)
        throw std::invalid_argument("embedding: vocab must be 2D");
    embed_shape.push_back(vocab->dim(0));
    TensorGraph::TensorNode* embed = vocab->graph()->data(
        std::move(embed_shape), output_name, vocab->dtype());

    embedding(index, vocab, embed, axis);
    return embed;
}

void embedding(TensorGraph::TensorNode* index,
               TensorGraph::TensorNode* vocab,
               TensorGraph::TensorNode* embed,
               Index axis)
{
    if(index == nullptr || vocab == nullptr || embed == nullptr)
        throw std::invalid_argument("embedding: tensors must be non-null");
    if(index->graph() != vocab->graph() || vocab->graph() != embed->graph())
        throw std::invalid_argument("embedding: tensors must belong to same graph");
    if(index->dtype() != DataType::INT64)
        throw std::invalid_argument("embedding: index must have INT64 dtype");
    if(vocab->dtype() != embed->dtype())
        throw std::invalid_argument("embedding: vocab and embed must have same dtype");
    validate_embedding_shape_and_merge(embed, index, vocab, "embedding");

    auto op = std::make_shared<TensorEmbeddingOp>(index, vocab, embed, axis);
    embed->graph()->add_op(op);
}

void TensorEmbeddingOp::execute(TensorGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(vocab);
    switch(dtype)
    {
        case DataType::FP32:
            run_embedding<nntile::fp32_t>(runtime, index, vocab, embed, axis);
            break;
        case DataType::FP32_FAST_TF32:
            run_embedding<nntile::fp32_fast_tf32_t>(runtime, index, vocab, embed, axis);
            break;
        case DataType::FP32_FAST_FP16:
            run_embedding<nntile::fp32_fast_fp16_t>(runtime, index, vocab, embed, axis);
            break;
        case DataType::FP32_FAST_BF16:
            run_embedding<nntile::fp32_fast_bf16_t>(runtime, index, vocab, embed, axis);
            break;
        case DataType::FP64:
            run_embedding<nntile::fp64_t>(runtime, index, vocab, embed, axis);
            break;
        case DataType::FP16:
            run_embedding<nntile::fp16_t>(runtime, index, vocab, embed, axis);
            break;
        case DataType::BF16:
            run_embedding<nntile::bf16_t>(runtime, index, vocab, embed, axis);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " not supported for embedding");
        default:
            throw std::runtime_error("Unsupported data type for embedding");
    }
}

} // namespace nntile::graph::tensor
