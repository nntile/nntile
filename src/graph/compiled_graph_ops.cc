/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/compiled_graph_ops.cc
 * Compiled graph operations.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/compiled_graph_ops.hh"

// Include standard headers
#include <stdexcept>

// Include other NNTile headers
#include "nntile/base_types.hh"
#include "nntile/tensor/add.hh"
#include "nntile/tensor/clear.hh"
#include "nntile/tensor/embedding.hh"
#include "nntile/tensor/embedding_backward.hh"
#include "nntile/tensor/gelu.hh"
#include "nntile/tensor/gelu_backward.hh"
#include "nntile/tensor/gelutanh.hh"
#include "nntile/tensor/gelutanh_backward.hh"
#include "nntile/tensor/gelutanh_inplace.hh"
#include "nntile/tensor/gelu_inplace.hh"
#include "nntile/tensor/gemm.hh"
#include "nntile/tensor/multiply.hh"
#include "nntile/tensor/relu.hh"
#include "nntile/tensor/relu_backward.hh"
#include "nntile/tensor/relu_inplace.hh"
#include "nntile/tensor/scale.hh"
#include "nntile/tensor/scale_inplace.hh"
#include "nntile/tensor/silu.hh"
#include "nntile/tensor/silu_backward.hh"
#include "nntile/tensor/silu_inplace.hh"
#include "nntile/tensor/sqrt.hh"
#include "nntile/tensor/sqrt_inplace.hh"
#include "nntile/tensor/sum.hh"
#include "nntile/tensor/sum_fiber.hh"

namespace nntile::graph
{

namespace
{

template<typename T>
void run_clear(CompiledGraph& graph, const std::string& name)
{
    auto& tensor = graph.get_tensor<T>(name);
    nntile::tensor::clear<T>(tensor);
}

template<typename T>
void run_gelu(CompiledGraph& graph,
              const std::string& x_name, const std::string& y_name)
{
    auto& x = graph.get_tensor<T>(x_name);
    auto& y = graph.get_tensor<T>(y_name);
    nntile::tensor::gelu<T>(x, y);
}

template<typename T>
void run_gelu_backward(CompiledGraph& graph,
                       const std::string& x_name, const std::string& dy_name,
                       const std::string& dx_name)
{
    auto& x = graph.get_tensor<T>(x_name);
    auto& dy = graph.get_tensor<T>(dy_name);
    auto& dx = graph.get_tensor<T>(dx_name);
    nntile::tensor::gelu_backward<T>(x, dy, dx);
}

template<typename T>
void run_gemm(CompiledGraph& graph, const GemmAttrs& attrs,
              const std::string& a_name, const std::string& b_name,
              const std::string& c_name)
{
    auto& a = graph.get_tensor<T>(a_name);
    auto& b = graph.get_tensor<T>(b_name);
    auto& c = graph.get_tensor<T>(c_name);

    const auto alpha = static_cast<nntile::Scalar>(attrs.alpha);
    const auto beta = static_cast<nntile::Scalar>(attrs.beta);
    const auto trans_a = attrs.trans_a ?
        nntile::TransOp(nntile::TransOp::Trans) :
        nntile::TransOp(nntile::TransOp::NoTrans);
    const auto trans_b = attrs.trans_b ?
        nntile::TransOp(nntile::TransOp::Trans) :
        nntile::TransOp(nntile::TransOp::NoTrans);

    nntile::tensor::gemm<T>(
        alpha,
        trans_a,
        a,
        trans_b,
        b,
        beta,
        c,
        attrs.ndim,
        attrs.batch_ndim,
        0  // redux = 0
    );
}

// Element-wise unary operations
template<typename T>
void run_gelu_inplace(CompiledGraph& graph, const std::string& x_name)
{
    auto& x = graph.get_tensor<T>(x_name);
    nntile::tensor::gelu_inplace<T>(x);
}

template<typename T>
void run_gelutanh(CompiledGraph& graph,
                  const std::string& x_name, const std::string& y_name)
{
    auto& x = graph.get_tensor<T>(x_name);
    auto& y = graph.get_tensor<T>(y_name);
    nntile::tensor::gelutanh<T>(x, y);
}

template<typename T>
void run_gelutanh_inplace(CompiledGraph& graph, const std::string& x_name)
{
    auto& x = graph.get_tensor<T>(x_name);
    nntile::tensor::gelutanh_inplace<T>(x);
}

template<typename T>
void run_gelutanh_backward(CompiledGraph& graph,
                           const std::string& x_name, const std::string& dy_name,
                           const std::string& dx_name)
{
    auto& x = graph.get_tensor<T>(x_name);
    auto& dy = graph.get_tensor<T>(dy_name);
    auto& dx = graph.get_tensor<T>(dx_name);
    nntile::tensor::gelutanh_backward<T>(x, dy, dx);
}

template<typename T>
void run_relu(CompiledGraph& graph,
              const std::string& x_name, const std::string& y_name)
{
    auto& x = graph.get_tensor<T>(x_name);
    auto& y = graph.get_tensor<T>(y_name);
    nntile::tensor::relu<T>(x, y);
}

template<typename T>
void run_relu_inplace(CompiledGraph& graph, const std::string& x_name)
{
    auto& x = graph.get_tensor<T>(x_name);
    nntile::tensor::relu_inplace<T>(x);
}

template<typename T>
void run_relu_backward(CompiledGraph& graph,
                       const std::string& x_name, const std::string& dy_name,
                       const std::string& dx_name)
{
    auto& x = graph.get_tensor<T>(x_name);
    auto& dy = graph.get_tensor<T>(dy_name);
    auto& dx = graph.get_tensor<T>(dx_name);
    nntile::tensor::relu_backward<T>(x, dy, dx);
}

template<typename T>
void run_silu(CompiledGraph& graph,
              const std::string& x_name, const std::string& y_name)
{
    auto& x = graph.get_tensor<T>(x_name);
    auto& y = graph.get_tensor<T>(y_name);
    nntile::tensor::silu<T>(x, y);
}

template<typename T>
void run_silu_inplace(CompiledGraph& graph, const std::string& x_name)
{
    auto& x = graph.get_tensor<T>(x_name);
    nntile::tensor::silu_inplace<T>(x);
}

template<typename T>
void run_silu_backward(CompiledGraph& graph,
                       const std::string& x_name, const std::string& dy_name,
                       const std::string& dx_name)
{
    auto& x = graph.get_tensor<T>(x_name);
    auto& dy = graph.get_tensor<T>(dy_name);
    auto& dx = graph.get_tensor<T>(dx_name);
    nntile::tensor::silu_backward<T>(x, dy, dx);
}

template<typename T>
void run_sqrt(CompiledGraph& graph,
              const std::string& x_name, const std::string& y_name)
{
    auto& x = graph.get_tensor<T>(x_name);
    auto& y = graph.get_tensor<T>(y_name);
    nntile::tensor::sqrt<T>(x, y);
}

template<typename T>
void run_sqrt_inplace(CompiledGraph& graph, const std::string& x_name)
{
    auto& x = graph.get_tensor<T>(x_name);
    nntile::tensor::sqrt_inplace<T>(x);
}

// Binary operations
template<typename T>
void run_add(CompiledGraph& graph, const BinaryOpAttrs& attrs,
             const std::string& x_name, const std::string& y_name,
             const std::string& z_name)
{
    auto& x = graph.get_tensor<T>(x_name);
    auto& y = graph.get_tensor<T>(y_name);
    auto& z = graph.get_tensor<T>(z_name);

    const auto alpha = static_cast<nntile::Scalar>(attrs.alpha);
    const auto beta = static_cast<nntile::Scalar>(attrs.beta);

    nntile::tensor::add<T>(alpha, x, beta, y, z);
}

template<typename T>
void run_add_inplace(CompiledGraph& graph, const BinaryOpAttrs& attrs,
                     const std::string& x_name, const std::string& y_name)
{
    auto& x = graph.get_tensor<T>(x_name);
    auto& y = graph.get_tensor<T>(y_name);

    const auto alpha = static_cast<nntile::Scalar>(attrs.alpha);
    const auto beta = static_cast<nntile::Scalar>(attrs.beta);

    nntile::tensor::add_inplace<T>(alpha, x, beta, y);
}

template<typename T>
void run_multiply(CompiledGraph& graph, const BinaryOpAttrs& attrs,
                  const std::string& x_name, const std::string& y_name,
                  const std::string& z_name)
{
    auto& x = graph.get_tensor<T>(x_name);
    auto& y = graph.get_tensor<T>(y_name);
    auto& z = graph.get_tensor<T>(z_name);

    nntile::tensor::multiply<T>(x, y, z);
}

template<typename T>
void run_multiply_inplace(CompiledGraph& graph, const BinaryOpAttrs& attrs,
                          const std::string& x_name, const std::string& y_name)
{
    auto& x = graph.get_tensor<T>(x_name);
    auto& y = graph.get_tensor<T>(y_name);

    nntile::tensor::multiply_inplace<T>(x, y);
}

// Reduction operations
template<typename T>
void run_sum(CompiledGraph& graph, const TotalSumAttrs& attrs,
             const std::string& x_name, const std::string& y_name)
{
    auto& x = graph.get_tensor<T>(x_name);
    auto& y = graph.get_tensor<T>(y_name);

    const auto alpha = static_cast<nntile::Scalar>(attrs.alpha);
    const auto beta = static_cast<nntile::Scalar>(attrs.beta);

    nntile::tensor::sum<T>(alpha, x, beta, y);
}

template<typename T>
void run_sum_fiber(CompiledGraph& graph, const ReductionAttrs& attrs,
                   const std::string& x_name, const std::string& y_name)
{
    auto& x = graph.get_tensor<T>(x_name);
    auto& y = graph.get_tensor<T>(y_name);

    const auto alpha = static_cast<nntile::Scalar>(attrs.alpha);
    const auto beta = static_cast<nntile::Scalar>(attrs.beta);

    nntile::tensor::sum_fiber<T>(alpha, x, beta, y,
                                  attrs.axis, attrs.batch_ndim, attrs.redux);
}

// Scale operations
template<typename T>
void run_scale(CompiledGraph& graph, const ScaleAttrs& attrs,
               const std::string& x_name, const std::string& y_name)
{
    auto& x = graph.get_tensor<T>(x_name);
    auto& y = graph.get_tensor<T>(y_name);

    const auto alpha = static_cast<nntile::Scalar>(attrs.alpha);

    nntile::tensor::scale<T>(alpha, x, y);
}

template<typename T>
void run_scale_inplace(CompiledGraph& graph, const ScaleAttrs& attrs,
                       const std::string& x_name)
{
    auto& x = graph.get_tensor<T>(x_name);

    const auto alpha = static_cast<nntile::Scalar>(attrs.alpha);

    nntile::tensor::scale_inplace<T>(alpha, x);
}

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

template<typename T>
void run_embedding_backward(CompiledGraph& graph, const EmbeddingAttrs& attrs,
                            const std::string& embed_name, const std::string& index_name,
                            const std::string& vocab_name)
{
    auto& embed = graph.get_tensor<T>(embed_name);
    auto& index = graph.get_tensor<int64_t>(index_name);
    auto& vocab = graph.get_tensor<T>(vocab_name);

    nntile::tensor::embedding_backward<T>(embed, index, vocab, attrs.axis);
}

} // namespace

//! Execute clear operation
void execute_clear(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const std::string& output_name = op_info.output_names[0];
    DataType dtype = graph.get_dtype(output_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_clear<nntile::fp32_t>(graph, output_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_clear<nntile::fp32_fast_tf32_t>(graph, output_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_clear<nntile::fp32_fast_fp16_t>(graph, output_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_clear<nntile::fp32_fast_bf16_t>(graph, output_name);
            break;
        case DataType::FP64:
            run_clear<nntile::fp64_t>(graph, output_name);
            break;
        case DataType::FP16:
            run_clear<nntile::fp16_t>(graph, output_name);
            break;
        case DataType::BF16:
            run_clear<nntile::bf16_t>(graph, output_name);
            break;
        case DataType::INT64:
            run_clear<nntile::int64_t>(graph, output_name);
            break;
        case DataType::BOOL:
            run_clear<nntile::bool_t>(graph, output_name);
            break;
        case DataType::INT32:
            throw std::runtime_error(
                "INT32 data type not supported for clear operation");
        default:
            throw std::runtime_error("Unsupported data type for clear");
    }
}

//! Execute gelu operation
void execute_gelu(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const std::string& x_name = op_info.input_names[0];
    const std::string& y_name = op_info.output_names[0];

    DataType dtype = graph.get_dtype(x_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_gelu<nntile::fp32_t>(graph, x_name, y_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_gelu<nntile::fp32_fast_tf32_t>(graph, x_name, y_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_gelu<nntile::fp32_fast_fp16_t>(graph, x_name, y_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_gelu<nntile::fp32_fast_bf16_t>(graph, x_name, y_name);
            break;
        case DataType::FP64:
            run_gelu<nntile::fp64_t>(graph, x_name, y_name);
            break;
        case DataType::FP16:
            run_gelu<nntile::fp16_t>(graph, x_name, y_name);
            break;
        case DataType::BF16:
            run_gelu<nntile::bf16_t>(graph, x_name, y_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for gelu operation");
        default:
            throw std::runtime_error("Unsupported data type for gelu");
    }
}

//! Execute gelu_backward operation
void execute_gelu_backward(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const std::string& x_name = op_info.input_names[0];
    const std::string& dy_name = op_info.input_names[1];
    const std::string& dx_name = op_info.input_names[2];  // dx is both input and output

    DataType dtype = graph.get_dtype(x_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_gelu_backward<nntile::fp32_t>(graph, x_name, dy_name, dx_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_gelu_backward<nntile::fp32_fast_tf32_t>(graph, x_name, dy_name, dx_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_gelu_backward<nntile::fp32_fast_fp16_t>(graph, x_name, dy_name, dx_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_gelu_backward<nntile::fp32_fast_bf16_t>(graph, x_name, dy_name, dx_name);
            break;
        case DataType::FP64:
            run_gelu_backward<nntile::fp64_t>(graph, x_name, dy_name, dx_name);
            break;
        case DataType::FP16:
            run_gelu_backward<nntile::fp16_t>(graph, x_name, dy_name, dx_name);
            break;
        case DataType::BF16:
            run_gelu_backward<nntile::bf16_t>(graph, x_name, dy_name, dx_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for gelu_backward operation");
        default:
            throw std::runtime_error("Unsupported data type for gelu_backward");
    }
}

//! Execute gemm operation
void execute_gemm(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const auto& attrs = std::get<GemmAttrs>(op_info.attrs);

    const std::string& a_name = op_info.input_names[0];
    const std::string& b_name = op_info.input_names[1];
    const std::string& c_name = op_info.output_names[0];

    DataType dtype = graph.get_dtype(a_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_gemm<nntile::fp32_t>(graph, attrs, a_name, b_name, c_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_gemm<nntile::fp32_fast_tf32_t>(graph, attrs, a_name, b_name, c_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_gemm<nntile::fp32_fast_fp16_t>(graph, attrs, a_name, b_name, c_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_gemm<nntile::fp32_fast_bf16_t>(graph, attrs, a_name, b_name, c_name);
            break;
        case DataType::FP64:
            run_gemm<nntile::fp64_t>(graph, attrs, a_name, b_name, c_name);
            break;
        case DataType::FP16:
            run_gemm<nntile::fp16_t>(graph, attrs, a_name, b_name, c_name);
            break;
        case DataType::BF16:
            run_gemm<nntile::bf16_t>(graph, attrs, a_name, b_name, c_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for gemm operation");
        default:
            throw std::runtime_error("Unsupported data type for gemm");
    }
}

//! Execute gelu_inplace operation
void execute_gelu_inplace(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const std::string& x_name = op_info.input_names[0];
    DataType dtype = graph.get_dtype(x_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_gelu_inplace<nntile::fp32_t>(graph, x_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_gelu_inplace<nntile::fp32_fast_tf32_t>(graph, x_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_gelu_inplace<nntile::fp32_fast_fp16_t>(graph, x_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_gelu_inplace<nntile::fp32_fast_bf16_t>(graph, x_name);
            break;
        case DataType::FP64:
            run_gelu_inplace<nntile::fp64_t>(graph, x_name);
            break;
        case DataType::FP16:
            run_gelu_inplace<nntile::fp16_t>(graph, x_name);
            break;
        case DataType::BF16:
            run_gelu_inplace<nntile::bf16_t>(graph, x_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for gelu_inplace operation");
        default:
            throw std::runtime_error("Unsupported data type for gelu_inplace");
    }
}

//! Execute add operation
void execute_add(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const BinaryOpAttrs& attrs = std::get<BinaryOpAttrs>(op_info.attrs);
    const std::string& x_name = op_info.input_names[0];
    const std::string& y_name = op_info.input_names[1];
    const std::string& z_name = op_info.output_names[0];
    DataType dtype = graph.get_dtype(x_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_add<nntile::fp32_t>(graph, attrs, x_name, y_name, z_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_add<nntile::fp32_fast_tf32_t>(graph, attrs, x_name, y_name, z_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_add<nntile::fp32_fast_fp16_t>(graph, attrs, x_name, y_name, z_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_add<nntile::fp32_fast_bf16_t>(graph, attrs, x_name, y_name, z_name);
            break;
        case DataType::FP64:
            run_add<nntile::fp64_t>(graph, attrs, x_name, y_name, z_name);
            break;
        case DataType::FP16:
            run_add<nntile::fp16_t>(graph, attrs, x_name, y_name, z_name);
            break;
        case DataType::BF16:
            run_add<nntile::bf16_t>(graph, attrs, x_name, y_name, z_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for add operation");
        default:
            throw std::runtime_error("Unsupported data type for add");
    }
}

//! Execute sum operation
void execute_sum(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const TotalSumAttrs& attrs = std::get<TotalSumAttrs>(op_info.attrs);
    const std::string& x_name = op_info.input_names[0];
    const std::string& y_name = op_info.input_names[1];  // Note: y is both input and output
    DataType dtype = graph.get_dtype(x_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_sum<nntile::fp32_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_sum<nntile::fp32_fast_tf32_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_sum<nntile::fp32_fast_fp16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_sum<nntile::fp32_fast_bf16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP64:
            run_sum<nntile::fp64_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP16:
            run_sum<nntile::fp16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::BF16:
            run_sum<nntile::bf16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for sum operation");
        default:
            throw std::runtime_error("Unsupported data type for sum");
    }
}

//! Execute embedding operation
void execute_embedding(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const EmbeddingAttrs& attrs = std::get<EmbeddingAttrs>(op_info.attrs);
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

// Add stub implementations for remaining operations
void execute_gelutanh(CompiledGraph& graph, const OpExecutionInfo& op_info) {
    throw std::runtime_error("execute_gelutanh not implemented yet");
}
void execute_gelutanh_inplace(CompiledGraph& graph, const OpExecutionInfo& op_info) {
    throw std::runtime_error("execute_gelutanh_inplace not implemented yet");
}
void execute_gelutanh_backward(CompiledGraph& graph, const OpExecutionInfo& op_info) {
    throw std::runtime_error("execute_gelutanh_backward not implemented yet");
}
void execute_relu(CompiledGraph& graph, const OpExecutionInfo& op_info) {
    throw std::runtime_error("execute_relu not implemented yet");
}
void execute_relu_inplace(CompiledGraph& graph, const OpExecutionInfo& op_info) {
    throw std::runtime_error("execute_relu_inplace not implemented yet");
}
void execute_relu_backward(CompiledGraph& graph, const OpExecutionInfo& op_info) {
    throw std::runtime_error("execute_relu_backward not implemented yet");
}
void execute_silu(CompiledGraph& graph, const OpExecutionInfo& op_info) {
    throw std::runtime_error("execute_silu not implemented yet");
}
void execute_silu_inplace(CompiledGraph& graph, const OpExecutionInfo& op_info) {
    throw std::runtime_error("execute_silu_inplace not implemented yet");
}
void execute_silu_backward(CompiledGraph& graph, const OpExecutionInfo& op_info) {
    throw std::runtime_error("execute_silu_backward not implemented yet");
}
void execute_sqrt(CompiledGraph& graph, const OpExecutionInfo& op_info) {
    throw std::runtime_error("execute_sqrt not implemented yet");
}
void execute_sqrt_inplace(CompiledGraph& graph, const OpExecutionInfo& op_info) {
    throw std::runtime_error("execute_sqrt_inplace not implemented yet");
}
void execute_add_inplace(CompiledGraph& graph, const OpExecutionInfo& op_info) {
    throw std::runtime_error("execute_add_inplace not implemented yet");
}
void execute_multiply(CompiledGraph& graph, const OpExecutionInfo& op_info) {
    throw std::runtime_error("execute_multiply not implemented yet");
}
void execute_multiply_inplace(CompiledGraph& graph, const OpExecutionInfo& op_info) {
    throw std::runtime_error("execute_multiply_inplace not implemented yet");
}
void execute_sum_fiber(CompiledGraph& graph, const OpExecutionInfo& op_info) {
    throw std::runtime_error("execute_sum_fiber not implemented yet");
}
void execute_scale(CompiledGraph& graph, const OpExecutionInfo& op_info) {
    throw std::runtime_error("execute_scale not implemented yet");
}
void execute_scale_inplace(CompiledGraph& graph, const OpExecutionInfo& op_info) {
    throw std::runtime_error("execute_scale_inplace not implemented yet");
}
void execute_embedding_backward(CompiledGraph& graph, const OpExecutionInfo& op_info) {
    throw std::runtime_error("execute_embedding_backward not implemented yet");
}

} // namespace nntile::graph
