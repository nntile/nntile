/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/compiled/embedding.cc
 * Compiled graph embedding operation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/compiled/embedding.hh"
#include "nntile/graph/logical/embedding.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/tensor/embedding.hh"

namespace nntile::graph
{

namespace
{

// Embedding operations
template<typename T>
void run_embedding(CompiledGraph& graph, const EmbeddingAttrs& attrs,
                   const std::string& index_name, const std::string& vocab_name,
                   const std::string& embed_name)
{
    auto& index = graph.get_tensor<int64_t>(index_name);
    auto& vocab = graph.get_tensor<T>(vocab_name);
    auto& embed = graph.get_tensor<T>(embed_name);

    nntile::tensor::embedding<T>(index, vocab, embed, attrs.axis);
}

} // namespace

//! Execute embedding operation
void execute_embedding(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const EmbeddingAttrs& attrs = *std::static_pointer_cast<EmbeddingAttrs>(op_info.attrs);
    const std::string& index_name = op_info.input_names[0];
    const std::string& vocab_name = op_info.input_names[1];
    const std::string& embed_name = op_info.output_names[0];
    DataType dtype = graph.get_dtype(vocab_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_embedding<nntile::fp32_t>(graph, attrs, index_name, vocab_name, embed_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_embedding<nntile::fp32_fast_tf32_t>(graph, attrs, index_name, vocab_name, embed_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_embedding<nntile::fp32_fast_fp16_t>(graph, attrs, index_name, vocab_name, embed_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_embedding<nntile::fp32_fast_bf16_t>(graph, attrs, index_name, vocab_name, embed_name);
            break;
        case DataType::FP64:
            run_embedding<nntile::fp64_t>(graph, attrs, index_name, vocab_name, embed_name);
            break;
        case DataType::FP16:
            run_embedding<nntile::fp16_t>(graph, attrs, index_name, vocab_name, embed_name);
            break;
        case DataType::BF16:
            run_embedding<nntile::bf16_t>(graph, attrs, index_name, vocab_name, embed_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for embedding operation");
        default:
            throw std::runtime_error("Unsupported data type for embedding");
    }
}

} // namespace nntile::graph
