/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/compiled/embedding_backward.cc
 * Compiled graph embedding_backward operation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/compiled/embedding_backward.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/tensor/embedding_backward.hh"

namespace nntile::graph
{

namespace
{

template<typename T>
void run_embedding_backward(CompiledGraph& graph, const EmbeddingAttrs& attrs,
                            const std::string& embed_name, const std::string& index_name,
                            const std::string& vocab_name)
{
    auto& embed = graph.get_tensor<T>(embed_name);
    auto& index = graph.get_tensor<int64_t>(index_name);
    auto& vocab = graph.get_tensor<T>(vocab_name);

    nntile::tensor::embedding_backward<T>(index, vocab, embed, attrs.axis);
}

} // namespace

void execute_embedding_backward(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const EmbeddingAttrs& attrs = std::get<EmbeddingAttrs>(op_info.attrs);
    const std::string& embed_name = op_info.input_names[0];
    const std::string& index_name = op_info.input_names[1];
    const std::string& vocab_name = op_info.input_names[2];
    DataType dtype = graph.get_dtype(embed_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_embedding_backward<nntile::fp32_t>(graph, attrs, embed_name, index_name, vocab_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_embedding_backward<nntile::fp32_fast_tf32_t>(graph, attrs, embed_name, index_name, vocab_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_embedding_backward<nntile::fp32_fast_fp16_t>(graph, attrs, embed_name, index_name, vocab_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_embedding_backward<nntile::fp32_fast_bf16_t>(graph, attrs, embed_name, index_name, vocab_name);
            break;
        case DataType::FP64:
            run_embedding_backward<nntile::fp64_t>(graph, attrs, embed_name, index_name, vocab_name);
            break;
        case DataType::FP16:
            run_embedding_backward<nntile::fp16_t>(graph, attrs, embed_name, index_name, vocab_name);
            break;
        case DataType::BF16:
            run_embedding_backward<nntile::bf16_t>(graph, attrs, embed_name, index_name, vocab_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for embedding_backward operation");
        default:
            throw std::runtime_error("Unsupported data type for embedding_backward");
    }
}

} // namespace nntile::graph
