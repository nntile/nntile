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

namespace nntile::graph
{

namespace
{

template<typename T>
void run_embedding_backward(TensorGraph::Runtime& runtime,
                           TensorGraph::TensorNode* index,
                           TensorGraph::TensorNode* embed,
                           TensorGraph::TensorNode* vocab,
                           Index axis,
                           int redux)
{
    auto& index_t = runtime.get_tensor<nntile::int64_t>(index);
    auto& embed_t = runtime.get_tensor<T>(embed);
    auto& vocab_t = runtime.get_tensor<T>(vocab);
    nntile::tensor::embedding_backward<T>(index_t, embed_t, vocab_t, axis, redux);
}

} // namespace

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
    auto op = std::make_shared<TensorEmbeddingBackwardOp>(
        index, embed, vocab, axis, redux);
    vocab->graph()->add_op(op);
}

void TensorEmbeddingBackwardOp::execute(TensorGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(embed);
    switch(dtype)
    {
        case DataType::FP32:
            run_embedding_backward<nntile::fp32_t>(
                runtime, index, embed, vocab, axis, redux);
            break;
        case DataType::FP32_FAST_TF32:
            run_embedding_backward<nntile::fp32_fast_tf32_t>(
                runtime, index, embed, vocab, axis, redux);
            break;
        case DataType::FP32_FAST_FP16:
            run_embedding_backward<nntile::fp32_fast_fp16_t>(
                runtime, index, embed, vocab, axis, redux);
            break;
        case DataType::FP32_FAST_BF16:
            run_embedding_backward<nntile::fp32_fast_bf16_t>(
                runtime, index, embed, vocab, axis, redux);
            break;
        case DataType::FP64:
            run_embedding_backward<nntile::fp64_t>(
                runtime, index, embed, vocab, axis, redux);
            break;
        case DataType::FP16:
            run_embedding_backward<nntile::fp16_t>(
                runtime, index, embed, vocab, axis, redux);
            break;
        case DataType::BF16:
            run_embedding_backward<nntile::bf16_t>(
                runtime, index, embed, vocab, axis, redux);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " not supported for embedding_backward");
        default:
            throw std::runtime_error("Unsupported data type for embedding_backward");
    }
}

} // namespace nntile::graph
