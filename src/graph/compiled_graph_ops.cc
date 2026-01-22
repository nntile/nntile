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
#include "nntile/tensor/add_inplace.hh"
#include "nntile/tensor/clear.hh"
#include "nntile/tensor/copy.hh"
#include "nntile/tensor/embedding.hh"
#include "nntile/tensor/embedding_backward.hh"
#include "nntile/tensor/fill.hh"
#include "nntile/tensor/gather.hh"
#include "nntile/tensor/gelu.hh"
#include "nntile/tensor/gelu_backward.hh"
#include "nntile/tensor/gelutanh.hh"
#include "nntile/tensor/gelutanh_backward.hh"
#include "nntile/tensor/gelutanh_inplace.hh"
#include "nntile/tensor/gelu_inplace.hh"
#include "nntile/tensor/gemm.hh"
#include "nntile/tensor/hypot.hh"
#include "nntile/tensor/hypot_inplace.hh"
#include "nntile/tensor/log_scalar.hh"
#include "nntile/tensor/logsumexp.hh"
#include "nntile/tensor/mask_scalar.hh"
#include "nntile/tensor/maxsumexp.hh"
#include "nntile/tensor/multiply.hh"
#include "nntile/tensor/multiply_inplace.hh"
#include "nntile/tensor/norm.hh"
#include "nntile/tensor/pow.hh"
#include "nntile/tensor/pow_inplace.hh"
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
#include "nntile/tensor/subtract_indexed_outputs.hh"
#include "nntile/tensor/sum.hh"
#include "nntile/tensor/sum_fiber.hh"
#include "nntile/tensor/sum_slice.hh"
#include "nntile/tensor/sumprod_fiber.hh"
#include "nntile/tensor/sumprod_slice.hh"
#include "nntile/tensor/transpose.hh"

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

    nntile::tensor::embedding_backward<T>(index, vocab, embed, attrs.axis);
}

// Element-wise operations
template<typename T>
void run_hypot(CompiledGraph& graph, const BinaryOpAttrs& attrs,
               const std::string& x_name, const std::string& y_name,
               const std::string& z_name)
{
    auto& x = graph.get_tensor<T>(x_name);
    auto& y = graph.get_tensor<T>(y_name);
    auto& z = graph.get_tensor<T>(z_name);

    const auto alpha = static_cast<nntile::Scalar>(attrs.alpha);
    const auto beta = static_cast<nntile::Scalar>(attrs.beta);

    nntile::tensor::hypot<T>(alpha, x, beta, y, z);
}

template<typename T>
void run_hypot_inplace(CompiledGraph& graph, const BinaryOpAttrs& attrs,
                       const std::string& x_name, const std::string& y_name)
{
    auto& x = graph.get_tensor<T>(x_name);
    auto& y = graph.get_tensor<T>(y_name);

    const auto alpha = static_cast<nntile::Scalar>(attrs.alpha);
    const auto beta = static_cast<nntile::Scalar>(attrs.beta);

    nntile::tensor::hypot_inplace<T>(alpha, x, beta, y);
}

template<typename T>
void run_pow(CompiledGraph& graph, const PowAttrs& attrs,
             const std::string& x_name, const std::string& y_name)
{
    auto& x = graph.get_tensor<T>(x_name);
    auto& y = graph.get_tensor<T>(y_name);

    const auto alpha = static_cast<nntile::Scalar>(attrs.alpha);
    const auto exp = static_cast<nntile::Scalar>(attrs.exponent);

    nntile::tensor::pow<T>(alpha, exp, x, y);
}

template<typename T>
void run_pow_inplace(CompiledGraph& graph, const PowAttrs& attrs,
                     const std::string& x_name)
{
    auto& x = graph.get_tensor<T>(x_name);

    const auto alpha = static_cast<nntile::Scalar>(attrs.alpha);
    const auto exp = static_cast<nntile::Scalar>(attrs.exponent);

    nntile::tensor::pow_inplace<T>(alpha, exp, x);
}

template<typename T>
void run_log_scalar(CompiledGraph& graph, const LogScalarAttrs& attrs,
                    const std::string& x_name)
{
    auto& x = graph.get_tensor<T>(x_name);

    // Note: LogScalarAttrs.base is not used in the current tensor operation
    // The tensor::log_scalar takes a name parameter
    nntile::tensor::log_scalar<T>("tensor_value", x);
}

template<typename T>
void run_mask_scalar(CompiledGraph& graph, const MaskScalarAttrs& attrs,
                     const std::string& mask_name, const std::string& x_name)
{
    auto& mask = graph.get_tensor<bool_t>(mask_name);
    auto& x = graph.get_tensor<T>(x_name);

    const auto val = static_cast<nntile::Scalar>(attrs.val);

    nntile::tensor::mask_scalar<T>(mask, val, x, attrs.batch_ndim);
}

// Reduction operations
template<typename T>
void run_sum_slice(CompiledGraph& graph, const ReductionAttrs& attrs,
                   const std::string& x_name, const std::string& y_name)
{
    auto& x = graph.get_tensor<T>(x_name);
    auto& y = graph.get_tensor<T>(y_name);

    const auto alpha = static_cast<nntile::Scalar>(attrs.alpha);
    const auto beta = static_cast<nntile::Scalar>(attrs.beta);

    nntile::tensor::sum_slice<T>(alpha, x, beta, y, attrs.axis, attrs.redux);
}

template<typename T>
void run_norm(CompiledGraph& graph, const TotalSumAttrs& attrs,
              const std::string& x_name, const std::string& y_name)
{
    auto& x = graph.get_tensor<T>(x_name);
    auto& y = graph.get_tensor<T>(y_name);

    const auto alpha = static_cast<nntile::Scalar>(attrs.alpha);
    const auto beta = static_cast<nntile::Scalar>(attrs.beta);

    nntile::tensor::norm<T>(alpha, x, beta, y);
}

template<typename T>
void run_logsumexp(CompiledGraph& graph, const LogSumExpAttrs& attrs,
                   const std::string& x_name, const std::string& y_name)
{
    auto& x = graph.get_tensor<T>(x_name);
    auto& y = graph.get_tensor<T>(y_name);

    nntile::tensor::logsumexp<T>(x, y);
}

template<typename T>
void run_maxsumexp(CompiledGraph& graph, const LogSumExpAttrs& attrs,
                   const std::string& x_name, const std::string& y_name)
{
    auto& x = graph.get_tensor<T>(x_name);
    auto& y = graph.get_tensor<T>(y_name);

    nntile::tensor::maxsumexp<T>(x, y, attrs.axis, 0);  // redux = 0
}

template<typename T>
void run_sumprod_fiber(CompiledGraph& graph, const ReductionAttrs& attrs,
                       const std::string& x1_name, const std::string& x2_name,
                       const std::string& y_name)
{
    auto& x1 = graph.get_tensor<T>(x1_name);
    auto& x2 = graph.get_tensor<T>(x2_name);
    auto& y = graph.get_tensor<T>(y_name);

    const auto alpha = static_cast<nntile::Scalar>(attrs.alpha);
    const auto beta = static_cast<nntile::Scalar>(attrs.beta);

    nntile::tensor::sumprod_fiber<T>(alpha, x1, x2, beta, y, attrs.axis, attrs.redux);
}

template<typename T>
void run_sumprod_slice(CompiledGraph& graph, const ReductionAttrs& attrs,
                       const std::string& x1_name, const std::string& x2_name,
                       const std::string& y_name)
{
    auto& x1 = graph.get_tensor<T>(x1_name);
    auto& x2 = graph.get_tensor<T>(x2_name);
    auto& y = graph.get_tensor<T>(y_name);

    const auto alpha = static_cast<nntile::Scalar>(attrs.alpha);
    const auto beta = static_cast<nntile::Scalar>(attrs.beta);

    nntile::tensor::sumprod_slice<T>(alpha, x1, x2, beta, y, attrs.axis, attrs.redux);
}

template<typename T>
void run_norm_fiber(CompiledGraph& graph, const ReductionAttrs& attrs,
                    const std::string& x_name, const std::string& y_name)
{
    auto& x = graph.get_tensor<T>(x_name);
    auto& y = graph.get_tensor<T>(y_name);

    const auto alpha = static_cast<nntile::Scalar>(attrs.alpha);
    const auto beta = static_cast<nntile::Scalar>(attrs.beta);

    nntile::tensor::norm_fiber<T>(alpha, x, beta, x, y, attrs.axis, attrs.batch_ndim, attrs.redux);
}

template<typename T>
void run_norm_fiber_inplace(CompiledGraph& graph, const ReductionAttrs& attrs,
                            const std::string& x_name, const std::string& y_name)
{
    auto& x = graph.get_tensor<T>(x_name);
    auto& y = graph.get_tensor<T>(y_name);

    const auto alpha = static_cast<nntile::Scalar>(attrs.alpha);
    const auto beta = static_cast<nntile::Scalar>(attrs.beta);

    nntile::tensor::norm_fiber_inplace<T>(alpha, x, beta, y, attrs.axis, attrs.batch_ndim, attrs.redux);
}

template<typename T>
void run_norm_slice(CompiledGraph& graph, const ReductionAttrs& attrs,
                    const std::string& x_name, const std::string& y_name)
{
    auto& x = graph.get_tensor<T>(x_name);
    auto& y = graph.get_tensor<T>(y_name);

    const auto alpha = static_cast<nntile::Scalar>(attrs.alpha);
    const auto beta = static_cast<nntile::Scalar>(attrs.beta);

    nntile::tensor::norm_slice<T>(alpha, x, beta, x, y, attrs.axis, attrs.redux);
}

template<typename T>
void run_norm_slice_inplace(CompiledGraph& graph, const ReductionAttrs& attrs,
                            const std::string& x_name, const std::string& y_name)
{
    auto& x = graph.get_tensor<T>(x_name);
    auto& y = graph.get_tensor<T>(y_name);

    const auto alpha = static_cast<nntile::Scalar>(attrs.alpha);
    const auto beta = static_cast<nntile::Scalar>(attrs.beta);

    nntile::tensor::norm_slice_inplace<T>(alpha, x, beta, y, attrs.axis, attrs.redux);
}

// Element-wise operations
template<typename T>
void run_hypot_scalar_inverse(CompiledGraph& graph, const HypotScalarInverseAttrs& attrs,
                              const std::string& x_name)
{
    auto& x = graph.get_tensor<T>(x_name);

    nntile::tensor::hypot_scalar_inverse<T>(attrs.eps, attrs.alpha, x);
}

template<typename T>
void run_subtract_indexed_outputs(CompiledGraph& graph, const SubtractIndexedOutputsAttrs& attrs,
                                  const std::string& labels_name, const std::string& x_name)
{
    auto& labels = graph.get_tensor<int64_t>(labels_name);
    auto& x = graph.get_tensor<T>(x_name);

    nntile::tensor::subtract_indexed_outputs<T>(attrs.val, labels, x, attrs.ignore_index);
}

// Utility operations
template<typename T>
void run_fill(CompiledGraph& graph, const FillAttrs& attrs,
              const std::string& x_name)
{
    auto& x = graph.get_tensor<T>(x_name);

    nntile::tensor::fill<T>(attrs.val, x);
}

template<typename T>
void run_copy(CompiledGraph& graph, const ClearAttrs& attrs,
              const std::string& x_name, const std::string& y_name)
{
    auto& x = graph.get_tensor<T>(x_name);
    auto& y = graph.get_tensor<T>(y_name);

    nntile::tensor::copy<T>(x, y);
}

template<typename T>
void run_transpose(CompiledGraph& graph, const TransposeAttrs& attrs,
                   const std::string& x_name, const std::string& y_name)
{
    auto& x = graph.get_tensor<T>(x_name);
    auto& y = graph.get_tensor<T>(y_name);

    nntile::tensor::transpose<T>(attrs.alpha, x, y, attrs.ndim);
}

template<typename T>
void run_gather(CompiledGraph& graph, const ClearAttrs& attrs,
                const std::string& x_name, const std::string& y_name)
{
    auto& x = graph.get_tensor<T>(x_name);
    auto& y = graph.get_tensor<T>(y_name);

    nntile::tensor::gather<T>(x, y);
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

//! Execute gelutanh operation
void execute_gelutanh(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const std::string& x_name = op_info.input_names[0];
    const std::string& y_name = op_info.output_names[0];

    DataType dtype = graph.get_dtype(x_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_gelutanh<nntile::fp32_t>(graph, x_name, y_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_gelutanh<nntile::fp32_fast_tf32_t>(graph, x_name, y_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_gelutanh<nntile::fp32_fast_fp16_t>(graph, x_name, y_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_gelutanh<nntile::fp32_fast_bf16_t>(graph, x_name, y_name);
            break;
        case DataType::FP64:
            run_gelutanh<nntile::fp64_t>(graph, x_name, y_name);
            break;
        case DataType::FP16:
            run_gelutanh<nntile::fp16_t>(graph, x_name, y_name);
            break;
        case DataType::BF16:
            run_gelutanh<nntile::bf16_t>(graph, x_name, y_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for gelutanh operation");
        default:
            throw std::runtime_error("Unsupported data type for gelutanh");
    }
}
//! Execute gelutanh_inplace operation
void execute_gelutanh_inplace(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const std::string& x_name = op_info.input_names[0];
    DataType dtype = graph.get_dtype(x_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_gelutanh_inplace<nntile::fp32_t>(graph, x_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_gelutanh_inplace<nntile::fp32_fast_tf32_t>(graph, x_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_gelutanh_inplace<nntile::fp32_fast_fp16_t>(graph, x_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_gelutanh_inplace<nntile::fp32_fast_bf16_t>(graph, x_name);
            break;
        case DataType::FP64:
            run_gelutanh_inplace<nntile::fp64_t>(graph, x_name);
            break;
        case DataType::FP16:
            run_gelutanh_inplace<nntile::fp16_t>(graph, x_name);
            break;
        case DataType::BF16:
            run_gelutanh_inplace<nntile::bf16_t>(graph, x_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for gelutanh_inplace operation");
        default:
            throw std::runtime_error("Unsupported data type for gelutanh_inplace");
    }
}
//! Execute gelutanh_backward operation
void execute_gelutanh_backward(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const std::string& x_name = op_info.input_names[0];
    const std::string& dy_name = op_info.input_names[1];
    const std::string& dx_name = op_info.input_names[2];  // dx is both input and output

    DataType dtype = graph.get_dtype(x_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_gelutanh_backward<nntile::fp32_t>(graph, x_name, dy_name, dx_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_gelutanh_backward<nntile::fp32_fast_tf32_t>(graph, x_name, dy_name, dx_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_gelutanh_backward<nntile::fp32_fast_fp16_t>(graph, x_name, dy_name, dx_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_gelutanh_backward<nntile::fp32_fast_bf16_t>(graph, x_name, dy_name, dx_name);
            break;
        case DataType::FP64:
            run_gelutanh_backward<nntile::fp64_t>(graph, x_name, dy_name, dx_name);
            break;
        case DataType::FP16:
            run_gelutanh_backward<nntile::fp16_t>(graph, x_name, dy_name, dx_name);
            break;
        case DataType::BF16:
            run_gelutanh_backward<nntile::bf16_t>(graph, x_name, dy_name, dx_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for gelutanh_backward operation");
        default:
            throw std::runtime_error("Unsupported data type for gelutanh_backward");
    }
}
//! Execute relu operation
void execute_relu(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const std::string& x_name = op_info.input_names[0];
    const std::string& y_name = op_info.output_names[0];

    DataType dtype = graph.get_dtype(x_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_relu<nntile::fp32_t>(graph, x_name, y_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_relu<nntile::fp32_fast_tf32_t>(graph, x_name, y_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_relu<nntile::fp32_fast_fp16_t>(graph, x_name, y_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_relu<nntile::fp32_fast_bf16_t>(graph, x_name, y_name);
            break;
        case DataType::FP64:
            run_relu<nntile::fp64_t>(graph, x_name, y_name);
            break;
        case DataType::FP16:
            run_relu<nntile::fp16_t>(graph, x_name, y_name);
            break;
        case DataType::BF16:
            run_relu<nntile::bf16_t>(graph, x_name, y_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for relu operation");
        default:
            throw std::runtime_error("Unsupported data type for relu");
    }
}
//! Execute relu_inplace operation
void execute_relu_inplace(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const std::string& x_name = op_info.input_names[0];
    DataType dtype = graph.get_dtype(x_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_relu_inplace<nntile::fp32_t>(graph, x_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_relu_inplace<nntile::fp32_fast_tf32_t>(graph, x_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_relu_inplace<nntile::fp32_fast_fp16_t>(graph, x_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_relu_inplace<nntile::fp32_fast_bf16_t>(graph, x_name);
            break;
        case DataType::FP64:
            run_relu_inplace<nntile::fp64_t>(graph, x_name);
            break;
        case DataType::FP16:
            run_relu_inplace<nntile::fp16_t>(graph, x_name);
            break;
        case DataType::BF16:
            run_relu_inplace<nntile::bf16_t>(graph, x_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for relu_inplace operation");
        default:
            throw std::runtime_error("Unsupported data type for relu_inplace");
    }
}
//! Execute relu_backward operation
void execute_relu_backward(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const std::string& x_name = op_info.input_names[0];
    const std::string& dy_name = op_info.input_names[1];
    const std::string& dx_name = op_info.input_names[2];  // dx is both input and output

    DataType dtype = graph.get_dtype(x_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_relu_backward<nntile::fp32_t>(graph, x_name, dy_name, dx_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_relu_backward<nntile::fp32_fast_tf32_t>(graph, x_name, dy_name, dx_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_relu_backward<nntile::fp32_fast_fp16_t>(graph, x_name, dy_name, dx_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_relu_backward<nntile::fp32_fast_bf16_t>(graph, x_name, dy_name, dx_name);
            break;
        case DataType::FP64:
            run_relu_backward<nntile::fp64_t>(graph, x_name, dy_name, dx_name);
            break;
        case DataType::FP16:
            run_relu_backward<nntile::fp16_t>(graph, x_name, dy_name, dx_name);
            break;
        case DataType::BF16:
            run_relu_backward<nntile::bf16_t>(graph, x_name, dy_name, dx_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for relu_backward operation");
        default:
            throw std::runtime_error("Unsupported data type for relu_backward");
    }
}
//! Execute silu operation
void execute_silu(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const std::string& x_name = op_info.input_names[0];
    const std::string& y_name = op_info.output_names[0];

    DataType dtype = graph.get_dtype(x_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_silu<nntile::fp32_t>(graph, x_name, y_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_silu<nntile::fp32_fast_tf32_t>(graph, x_name, y_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_silu<nntile::fp32_fast_fp16_t>(graph, x_name, y_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_silu<nntile::fp32_fast_bf16_t>(graph, x_name, y_name);
            break;
        case DataType::FP64:
            run_silu<nntile::fp64_t>(graph, x_name, y_name);
            break;
        case DataType::FP16:
            run_silu<nntile::fp16_t>(graph, x_name, y_name);
            break;
        case DataType::BF16:
            run_silu<nntile::bf16_t>(graph, x_name, y_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for silu operation");
        default:
            throw std::runtime_error("Unsupported data type for silu");
    }
}
//! Execute silu_inplace operation
void execute_silu_inplace(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const std::string& x_name = op_info.input_names[0];
    DataType dtype = graph.get_dtype(x_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_silu_inplace<nntile::fp32_t>(graph, x_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_silu_inplace<nntile::fp32_fast_tf32_t>(graph, x_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_silu_inplace<nntile::fp32_fast_fp16_t>(graph, x_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_silu_inplace<nntile::fp32_fast_bf16_t>(graph, x_name);
            break;
        case DataType::FP64:
            run_silu_inplace<nntile::fp64_t>(graph, x_name);
            break;
        case DataType::FP16:
            run_silu_inplace<nntile::fp16_t>(graph, x_name);
            break;
        case DataType::BF16:
            run_silu_inplace<nntile::bf16_t>(graph, x_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for silu_inplace operation");
        default:
            throw std::runtime_error("Unsupported data type for silu_inplace");
    }
}
//! Execute silu_backward operation
void execute_silu_backward(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const std::string& x_name = op_info.input_names[0];
    const std::string& dy_name = op_info.input_names[1];
    const std::string& dx_name = op_info.input_names[2];  // dx is both input and output

    DataType dtype = graph.get_dtype(x_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_silu_backward<nntile::fp32_t>(graph, x_name, dy_name, dx_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_silu_backward<nntile::fp32_fast_tf32_t>(graph, x_name, dy_name, dx_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_silu_backward<nntile::fp32_fast_fp16_t>(graph, x_name, dy_name, dx_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_silu_backward<nntile::fp32_fast_bf16_t>(graph, x_name, dy_name, dx_name);
            break;
        case DataType::FP64:
            run_silu_backward<nntile::fp64_t>(graph, x_name, dy_name, dx_name);
            break;
        case DataType::FP16:
            run_silu_backward<nntile::fp16_t>(graph, x_name, dy_name, dx_name);
            break;
        case DataType::BF16:
            run_silu_backward<nntile::bf16_t>(graph, x_name, dy_name, dx_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for silu_backward operation");
        default:
            throw std::runtime_error("Unsupported data type for silu_backward");
    }
}
//! Execute sqrt operation
void execute_sqrt(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const std::string& x_name = op_info.input_names[0];
    const std::string& y_name = op_info.output_names[0];

    DataType dtype = graph.get_dtype(x_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_sqrt<nntile::fp32_t>(graph, x_name, y_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_sqrt<nntile::fp32_fast_tf32_t>(graph, x_name, y_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_sqrt<nntile::fp32_fast_fp16_t>(graph, x_name, y_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_sqrt<nntile::fp32_fast_bf16_t>(graph, x_name, y_name);
            break;
        case DataType::FP64:
            run_sqrt<nntile::fp64_t>(graph, x_name, y_name);
            break;
        case DataType::FP16:
            run_sqrt<nntile::fp16_t>(graph, x_name, y_name);
            break;
        case DataType::BF16:
            run_sqrt<nntile::bf16_t>(graph, x_name, y_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for sqrt operation");
        default:
            throw std::runtime_error("Unsupported data type for sqrt");
    }
}
//! Execute sqrt_inplace operation
void execute_sqrt_inplace(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const std::string& x_name = op_info.input_names[0];
    DataType dtype = graph.get_dtype(x_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_sqrt_inplace<nntile::fp32_t>(graph, x_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_sqrt_inplace<nntile::fp32_fast_tf32_t>(graph, x_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_sqrt_inplace<nntile::fp32_fast_fp16_t>(graph, x_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_sqrt_inplace<nntile::fp32_fast_bf16_t>(graph, x_name);
            break;
        case DataType::FP64:
            run_sqrt_inplace<nntile::fp64_t>(graph, x_name);
            break;
        case DataType::FP16:
            run_sqrt_inplace<nntile::fp16_t>(graph, x_name);
            break;
        case DataType::BF16:
            run_sqrt_inplace<nntile::bf16_t>(graph, x_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for sqrt_inplace operation");
        default:
            throw std::runtime_error("Unsupported data type for sqrt_inplace");
    }
}
//! Execute add_inplace operation
void execute_add_inplace(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const BinaryOpAttrs& attrs = std::get<BinaryOpAttrs>(op_info.attrs);
    const std::string& x_name = op_info.input_names[0];
    const std::string& y_name = op_info.input_names[1];  // y is both input and output
    DataType dtype = graph.get_dtype(x_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_add_inplace<nntile::fp32_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_add_inplace<nntile::fp32_fast_tf32_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_add_inplace<nntile::fp32_fast_fp16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_add_inplace<nntile::fp32_fast_bf16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP64:
            run_add_inplace<nntile::fp64_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP16:
            run_add_inplace<nntile::fp16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::BF16:
            run_add_inplace<nntile::bf16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for add_inplace operation");
        default:
            throw std::runtime_error("Unsupported data type for add_inplace");
    }
}
//! Execute multiply operation
void execute_multiply(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const BinaryOpAttrs& attrs = std::get<BinaryOpAttrs>(op_info.attrs);
    const std::string& x_name = op_info.input_names[0];
    const std::string& y_name = op_info.input_names[1];
    const std::string& z_name = op_info.output_names[0];
    DataType dtype = graph.get_dtype(x_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_multiply<nntile::fp32_t>(graph, attrs, x_name, y_name, z_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_multiply<nntile::fp32_fast_tf32_t>(graph, attrs, x_name, y_name, z_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_multiply<nntile::fp32_fast_fp16_t>(graph, attrs, x_name, y_name, z_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_multiply<nntile::fp32_fast_bf16_t>(graph, attrs, x_name, y_name, z_name);
            break;
        case DataType::FP64:
            run_multiply<nntile::fp64_t>(graph, attrs, x_name, y_name, z_name);
            break;
        case DataType::FP16:
            run_multiply<nntile::fp16_t>(graph, attrs, x_name, y_name, z_name);
            break;
        case DataType::BF16:
            run_multiply<nntile::bf16_t>(graph, attrs, x_name, y_name, z_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for multiply operation");
        default:
            throw std::runtime_error("Unsupported data type for multiply");
    }
}
//! Execute multiply_inplace operation
void execute_multiply_inplace(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const BinaryOpAttrs& attrs = std::get<BinaryOpAttrs>(op_info.attrs);
    const std::string& x_name = op_info.input_names[0];
    const std::string& y_name = op_info.input_names[1];  // y is both input and output
    DataType dtype = graph.get_dtype(x_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_multiply_inplace<nntile::fp32_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_multiply_inplace<nntile::fp32_fast_tf32_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_multiply_inplace<nntile::fp32_fast_fp16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_multiply_inplace<nntile::fp32_fast_bf16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP64:
            run_multiply_inplace<nntile::fp64_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP16:
            run_multiply_inplace<nntile::fp16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::BF16:
            run_multiply_inplace<nntile::bf16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for multiply_inplace operation");
        default:
            throw std::runtime_error("Unsupported data type for multiply_inplace");
    }
}
//! Execute sum_fiber operation
void execute_sum_fiber(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const ReductionAttrs& attrs = std::get<ReductionAttrs>(op_info.attrs);
    const std::string& x_name = op_info.input_names[0];
    const std::string& y_name = op_info.input_names[1];  // y is both input and output
    DataType dtype = graph.get_dtype(x_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_sum_fiber<nntile::fp32_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_sum_fiber<nntile::fp32_fast_tf32_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_sum_fiber<nntile::fp32_fast_fp16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_sum_fiber<nntile::fp32_fast_bf16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP64:
            run_sum_fiber<nntile::fp64_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP16:
            run_sum_fiber<nntile::fp16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::BF16:
            run_sum_fiber<nntile::bf16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for sum_fiber operation");
        default:
            throw std::runtime_error("Unsupported data type for sum_fiber");
    }
}
//! Execute scale operation
void execute_scale(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const ScaleAttrs& attrs = std::get<ScaleAttrs>(op_info.attrs);
    const std::string& x_name = op_info.input_names[0];
    const std::string& y_name = op_info.output_names[0];
    DataType dtype = graph.get_dtype(x_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_scale<nntile::fp32_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_scale<nntile::fp32_fast_tf32_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_scale<nntile::fp32_fast_fp16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_scale<nntile::fp32_fast_bf16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP64:
            run_scale<nntile::fp64_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP16:
            run_scale<nntile::fp16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::BF16:
            run_scale<nntile::bf16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for scale operation");
        default:
            throw std::runtime_error("Unsupported data type for scale");
    }
}
//! Execute scale_inplace operation
void execute_scale_inplace(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const ScaleAttrs& attrs = std::get<ScaleAttrs>(op_info.attrs);
    const std::string& x_name = op_info.input_names[0];
    DataType dtype = graph.get_dtype(x_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_scale_inplace<nntile::fp32_t>(graph, attrs, x_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_scale_inplace<nntile::fp32_fast_tf32_t>(graph, attrs, x_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_scale_inplace<nntile::fp32_fast_fp16_t>(graph, attrs, x_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_scale_inplace<nntile::fp32_fast_bf16_t>(graph, attrs, x_name);
            break;
        case DataType::FP64:
            run_scale_inplace<nntile::fp64_t>(graph, attrs, x_name);
            break;
        case DataType::FP16:
            run_scale_inplace<nntile::fp16_t>(graph, attrs, x_name);
            break;
        case DataType::BF16:
            run_scale_inplace<nntile::bf16_t>(graph, attrs, x_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for scale_inplace operation");
        default:
            throw std::runtime_error("Unsupported data type for scale_inplace");
    }
}
//! Execute embedding_backward operation
void execute_embedding_backward(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const EmbeddingAttrs& attrs = std::get<EmbeddingAttrs>(op_info.attrs);
    const std::string& embed_name = op_info.input_names[0];
    const std::string& index_name = op_info.input_names[1];
    const std::string& vocab_name = op_info.input_names[2];  // vocab is both input and output
    DataType dtype = graph.get_dtype(vocab_name);

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

//! Execute hypot operation
void execute_hypot(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const BinaryOpAttrs& attrs = std::get<BinaryOpAttrs>(op_info.attrs);
    const std::string& x_name = op_info.input_names[0];
    const std::string& y_name = op_info.input_names[1];
    const std::string& z_name = op_info.output_names[0];
    DataType dtype = graph.get_dtype(x_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_hypot<nntile::fp32_t>(graph, attrs, x_name, y_name, z_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_hypot<nntile::fp32_fast_tf32_t>(graph, attrs, x_name, y_name, z_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_hypot<nntile::fp32_fast_fp16_t>(graph, attrs, x_name, y_name, z_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_hypot<nntile::fp32_fast_bf16_t>(graph, attrs, x_name, y_name, z_name);
            break;
        case DataType::FP64:
            run_hypot<nntile::fp64_t>(graph, attrs, x_name, y_name, z_name);
            break;
        case DataType::FP16:
            run_hypot<nntile::fp16_t>(graph, attrs, x_name, y_name, z_name);
            break;
        case DataType::BF16:
            run_hypot<nntile::bf16_t>(graph, attrs, x_name, y_name, z_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for hypot operation");
        default:
            throw std::runtime_error("Unsupported data type for hypot");
    }
}

//! Execute hypot_inplace operation
void execute_hypot_inplace(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const BinaryOpAttrs& attrs = std::get<BinaryOpAttrs>(op_info.attrs);
    const std::string& x_name = op_info.input_names[0];
    const std::string& y_name = op_info.input_names[1];  // y is both input and output
    DataType dtype = graph.get_dtype(x_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_hypot_inplace<nntile::fp32_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_hypot_inplace<nntile::fp32_fast_tf32_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_hypot_inplace<nntile::fp32_fast_fp16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_hypot_inplace<nntile::fp32_fast_bf16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP64:
            run_hypot_inplace<nntile::fp64_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP16:
            run_hypot_inplace<nntile::fp16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::BF16:
            run_hypot_inplace<nntile::bf16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for hypot_inplace operation");
        default:
            throw std::runtime_error("Unsupported data type for hypot_inplace");
    }
}

//! Execute pow operation
void execute_pow(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const PowAttrs& attrs = std::get<PowAttrs>(op_info.attrs);
    const std::string& x_name = op_info.input_names[0];
    const std::string& y_name = op_info.output_names[0];
    DataType dtype = graph.get_dtype(x_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_pow<nntile::fp32_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_pow<nntile::fp32_fast_tf32_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_pow<nntile::fp32_fast_fp16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_pow<nntile::fp32_fast_bf16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP64:
            run_pow<nntile::fp64_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP16:
            run_pow<nntile::fp16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::BF16:
            run_pow<nntile::bf16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for pow operation");
        default:
            throw std::runtime_error("Unsupported data type for pow");
    }
}

//! Execute pow_inplace operation
void execute_pow_inplace(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const PowAttrs& attrs = std::get<PowAttrs>(op_info.attrs);
    const std::string& x_name = op_info.input_names[0];
    DataType dtype = graph.get_dtype(x_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_pow_inplace<nntile::fp32_t>(graph, attrs, x_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_pow_inplace<nntile::fp32_fast_tf32_t>(graph, attrs, x_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_pow_inplace<nntile::fp32_fast_fp16_t>(graph, attrs, x_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_pow_inplace<nntile::fp32_fast_bf16_t>(graph, attrs, x_name);
            break;
        case DataType::FP64:
            run_pow_inplace<nntile::fp64_t>(graph, attrs, x_name);
            break;
        case DataType::FP16:
            run_pow_inplace<nntile::fp16_t>(graph, attrs, x_name);
            break;
        case DataType::BF16:
            run_pow_inplace<nntile::bf16_t>(graph, attrs, x_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for pow_inplace operation");
        default:
            throw std::runtime_error("Unsupported data type for pow_inplace");
    }
}

//! Execute log_scalar operation
void execute_log_scalar(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const LogScalarAttrs& attrs = std::get<LogScalarAttrs>(op_info.attrs);
    const std::string& x_name = op_info.input_names[0];
    DataType dtype = graph.get_dtype(x_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_log_scalar<nntile::fp32_t>(graph, attrs, x_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_log_scalar<nntile::fp32_fast_tf32_t>(graph, attrs, x_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_log_scalar<nntile::fp32_fast_fp16_t>(graph, attrs, x_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_log_scalar<nntile::fp32_fast_bf16_t>(graph, attrs, x_name);
            break;
        case DataType::FP64:
            run_log_scalar<nntile::fp64_t>(graph, attrs, x_name);
            break;
        case DataType::FP16:
            run_log_scalar<nntile::fp16_t>(graph, attrs, x_name);
            break;
        case DataType::BF16:
            run_log_scalar<nntile::bf16_t>(graph, attrs, x_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for log_scalar operation");
        default:
            throw std::runtime_error("Unsupported data type for log_scalar");
    }
}

//! Execute mask_scalar operation
void execute_mask_scalar(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const MaskScalarAttrs& attrs = std::get<MaskScalarAttrs>(op_info.attrs);
    const std::string& mask_name = op_info.input_names[0];
    const std::string& x_name = op_info.input_names[1];  // x is both input and output
    DataType dtype = graph.get_dtype(x_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_mask_scalar<nntile::fp32_t>(graph, attrs, mask_name, x_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_mask_scalar<nntile::fp32_fast_tf32_t>(graph, attrs, mask_name, x_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_mask_scalar<nntile::fp32_fast_fp16_t>(graph, attrs, mask_name, x_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_mask_scalar<nntile::fp32_fast_bf16_t>(graph, attrs, mask_name, x_name);
            break;
        case DataType::FP64:
            run_mask_scalar<nntile::fp64_t>(graph, attrs, mask_name, x_name);
            break;
        case DataType::FP16:
            run_mask_scalar<nntile::fp16_t>(graph, attrs, mask_name, x_name);
            break;
        case DataType::BF16:
            run_mask_scalar<nntile::bf16_t>(graph, attrs, mask_name, x_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for mask_scalar operation");
        default:
            throw std::runtime_error("Unsupported data type for mask_scalar");
    }
}

//! Execute sum_slice operation
void execute_sum_slice(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const ReductionAttrs& attrs = std::get<ReductionAttrs>(op_info.attrs);
    const std::string& x_name = op_info.input_names[0];
    const std::string& y_name = op_info.input_names[1];  // y is both input and output
    DataType dtype = graph.get_dtype(x_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_sum_slice<nntile::fp32_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_sum_slice<nntile::fp32_fast_tf32_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_sum_slice<nntile::fp32_fast_fp16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_sum_slice<nntile::fp32_fast_bf16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP64:
            run_sum_slice<nntile::fp64_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP16:
            run_sum_slice<nntile::fp16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::BF16:
            run_sum_slice<nntile::bf16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for sum_slice operation");
        default:
            throw std::runtime_error("Unsupported data type for sum_slice");
    }
}

//! Execute norm operation
void execute_norm(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const TotalSumAttrs& attrs = std::get<TotalSumAttrs>(op_info.attrs);
    const std::string& x_name = op_info.input_names[0];
    const std::string& y_name = op_info.input_names[1];  // y is both input and output
    DataType dtype = graph.get_dtype(x_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_norm<nntile::fp32_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_norm<nntile::fp32_fast_tf32_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_norm<nntile::fp32_fast_fp16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_norm<nntile::fp32_fast_bf16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP64:
            run_norm<nntile::fp64_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP16:
            run_norm<nntile::fp16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::BF16:
            run_norm<nntile::bf16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for norm operation");
        default:
            throw std::runtime_error("Unsupported data type for norm");
    }
}

//! Execute logsumexp operation
void execute_logsumexp(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const LogSumExpAttrs& attrs = std::get<LogSumExpAttrs>(op_info.attrs);
    const std::string& x_name = op_info.input_names[0];
    const std::string& y_name = op_info.output_names[0];
    DataType dtype = graph.get_dtype(x_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_logsumexp<nntile::fp32_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_logsumexp<nntile::fp32_fast_tf32_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_logsumexp<nntile::fp32_fast_fp16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_logsumexp<nntile::fp32_fast_bf16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP64:
            run_logsumexp<nntile::fp64_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP16:
            run_logsumexp<nntile::fp16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::BF16:
            run_logsumexp<nntile::bf16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for logsumexp operation");
        default:
            throw std::runtime_error("Unsupported data type for logsumexp");
    }
}

//! Execute maxsumexp operation
void execute_maxsumexp(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const LogSumExpAttrs& attrs = std::get<LogSumExpAttrs>(op_info.attrs);
    const std::string& x_name = op_info.input_names[0];
    const std::string& y_name = op_info.output_names[0];
    DataType dtype = graph.get_dtype(x_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_maxsumexp<nntile::fp32_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_maxsumexp<nntile::fp32_fast_tf32_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_maxsumexp<nntile::fp32_fast_fp16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_maxsumexp<nntile::fp32_fast_bf16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP64:
            run_maxsumexp<nntile::fp64_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP16:
            run_maxsumexp<nntile::fp16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::BF16:
            run_maxsumexp<nntile::bf16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for maxsumexp operation");
        default:
            throw std::runtime_error("Unsupported data type for maxsumexp");
    }
}

//! Execute sumprod_fiber operation
void execute_sumprod_fiber(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const ReductionAttrs& attrs = std::get<ReductionAttrs>(op_info.attrs);
    const std::string& x1_name = op_info.input_names[0];
    const std::string& x2_name = op_info.input_names[1];
    const std::string& y_name = op_info.input_names[2];  // y is both input and output
    DataType dtype = graph.get_dtype(x1_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_sumprod_fiber<nntile::fp32_t>(graph, attrs, x1_name, x2_name, y_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_sumprod_fiber<nntile::fp32_fast_tf32_t>(graph, attrs, x1_name, x2_name, y_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_sumprod_fiber<nntile::fp32_fast_fp16_t>(graph, attrs, x1_name, x2_name, y_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_sumprod_fiber<nntile::fp32_fast_bf16_t>(graph, attrs, x1_name, x2_name, y_name);
            break;
        case DataType::FP64:
            run_sumprod_fiber<nntile::fp64_t>(graph, attrs, x1_name, x2_name, y_name);
            break;
        case DataType::FP16:
            run_sumprod_fiber<nntile::fp16_t>(graph, attrs, x1_name, x2_name, y_name);
            break;
        case DataType::BF16:
            run_sumprod_fiber<nntile::bf16_t>(graph, attrs, x1_name, x2_name, y_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for sumprod_fiber operation");
        default:
            throw std::runtime_error("Unsupported data type for sumprod_fiber");
    }
}

//! Execute sumprod_slice operation
void execute_sumprod_slice(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const ReductionAttrs& attrs = std::get<ReductionAttrs>(op_info.attrs);
    const std::string& x1_name = op_info.input_names[0];
    const std::string& x2_name = op_info.input_names[1];
    const std::string& y_name = op_info.input_names[2];  // y is both input and output
    DataType dtype = graph.get_dtype(x1_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_sumprod_slice<nntile::fp32_t>(graph, attrs, x1_name, x2_name, y_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_sumprod_slice<nntile::fp32_fast_tf32_t>(graph, attrs, x1_name, x2_name, y_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_sumprod_slice<nntile::fp32_fast_fp16_t>(graph, attrs, x1_name, x2_name, y_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_sumprod_slice<nntile::fp32_fast_bf16_t>(graph, attrs, x1_name, x2_name, y_name);
            break;
        case DataType::FP64:
            run_sumprod_slice<nntile::fp64_t>(graph, attrs, x1_name, x2_name, y_name);
            break;
        case DataType::FP16:
            run_sumprod_slice<nntile::fp16_t>(graph, attrs, x1_name, x2_name, y_name);
            break;
        case DataType::BF16:
            run_sumprod_slice<nntile::bf16_t>(graph, attrs, x1_name, x2_name, y_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for sumprod_slice operation");
        default:
            throw std::runtime_error("Unsupported data type for sumprod_slice");
    }
}

//! Execute norm_fiber operation
void execute_norm_fiber(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const ReductionAttrs& attrs = std::get<ReductionAttrs>(op_info.attrs);
    const std::string& x_name = op_info.input_names[0];
    const std::string& y_name = op_info.input_names[1];  // y is both input and output
    DataType dtype = graph.get_dtype(x_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_norm_fiber<nntile::fp32_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_norm_fiber<nntile::fp32_fast_tf32_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_norm_fiber<nntile::fp32_fast_fp16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_norm_fiber<nntile::fp32_fast_bf16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP64:
            run_norm_fiber<nntile::fp64_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP16:
            run_norm_fiber<nntile::fp16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::BF16:
            run_norm_fiber<nntile::bf16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for norm_fiber operation");
        default:
            throw std::runtime_error("Unsupported data type for norm_fiber");
    }
}

//! Execute norm_fiber_inplace operation
void execute_norm_fiber_inplace(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const ReductionAttrs& attrs = std::get<ReductionAttrs>(op_info.attrs);
    const std::string& x_name = op_info.input_names[0];
    const std::string& y_name = op_info.input_names[1];  // y is both input and output
    DataType dtype = graph.get_dtype(x_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_norm_fiber_inplace<nntile::fp32_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_norm_fiber_inplace<nntile::fp32_fast_tf32_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_norm_fiber_inplace<nntile::fp32_fast_fp16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_norm_fiber_inplace<nntile::fp32_fast_bf16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP64:
            run_norm_fiber_inplace<nntile::fp64_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP16:
            run_norm_fiber_inplace<nntile::fp16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::BF16:
            run_norm_fiber_inplace<nntile::bf16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for norm_fiber_inplace operation");
        default:
            throw std::runtime_error("Unsupported data type for norm_fiber_inplace");
    }
}

//! Execute norm_slice operation
void execute_norm_slice(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const ReductionAttrs& attrs = std::get<ReductionAttrs>(op_info.attrs);
    const std::string& x_name = op_info.input_names[0];
    const std::string& y_name = op_info.input_names[1];  // y is both input and output
    DataType dtype = graph.get_dtype(x_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_norm_slice<nntile::fp32_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_norm_slice<nntile::fp32_fast_tf32_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_norm_slice<nntile::fp32_fast_fp16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_norm_slice<nntile::fp32_fast_bf16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP64:
            run_norm_slice<nntile::fp64_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP16:
            run_norm_slice<nntile::fp16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::BF16:
            run_norm_slice<nntile::bf16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for norm_slice operation");
        default:
            throw std::runtime_error("Unsupported data type for norm_slice");
    }
}

//! Execute norm_slice_inplace operation
void execute_norm_slice_inplace(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const ReductionAttrs& attrs = std::get<ReductionAttrs>(op_info.attrs);
    const std::string& x_name = op_info.input_names[0];
    const std::string& y_name = op_info.input_names[1];  // y is both input and output
    DataType dtype = graph.get_dtype(x_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_norm_slice_inplace<nntile::fp32_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_norm_slice_inplace<nntile::fp32_fast_tf32_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_norm_slice_inplace<nntile::fp32_fast_fp16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_norm_slice_inplace<nntile::fp32_fast_bf16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP64:
            run_norm_slice_inplace<nntile::fp64_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP16:
            run_norm_slice_inplace<nntile::fp16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::BF16:
            run_norm_slice_inplace<nntile::bf16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for norm_slice_inplace operation");
        default:
            throw std::runtime_error("Unsupported data type for norm_slice_inplace");
    }
}

//! Execute fill operation
void execute_fill(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const FillAttrs& attrs = std::get<FillAttrs>(op_info.attrs);
    const std::string& x_name = op_info.output_names[0];
    DataType dtype = graph.get_dtype(x_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_fill<nntile::fp32_t>(graph, attrs, x_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_fill<nntile::fp32_fast_tf32_t>(graph, attrs, x_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_fill<nntile::fp32_fast_fp16_t>(graph, attrs, x_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_fill<nntile::fp32_fast_bf16_t>(graph, attrs, x_name);
            break;
        case DataType::FP64:
            run_fill<nntile::fp64_t>(graph, attrs, x_name);
            break;
        case DataType::FP16:
            run_fill<nntile::fp16_t>(graph, attrs, x_name);
            break;
        case DataType::BF16:
            run_fill<nntile::bf16_t>(graph, attrs, x_name);
            break;
        case DataType::INT64:
            run_fill<nntile::int64_t>(graph, attrs, x_name);
            break;
        case DataType::INT32:
            run_fill<nntile::int32_t>(graph, attrs, x_name);
            break;
        case DataType::BOOL:
            run_fill<nntile::bool_t>(graph, attrs, x_name);
            break;
        default:
            throw std::runtime_error("Unsupported data type for fill");
    }
}

//! Execute copy operation
void execute_copy(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const ClearAttrs& attrs = std::get<ClearAttrs>(op_info.attrs);
    const std::string& x_name = op_info.input_names[0];
    const std::string& y_name = op_info.output_names[0];
    DataType dtype = graph.get_dtype(x_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_copy<nntile::fp32_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_copy<nntile::fp32_fast_tf32_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_copy<nntile::fp32_fast_fp16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_copy<nntile::fp32_fast_bf16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP64:
            run_copy<nntile::fp64_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP16:
            run_copy<nntile::fp16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::BF16:
            run_copy<nntile::bf16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::INT64:
            run_copy<nntile::int64_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::INT32:
            run_copy<nntile::int32_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::BOOL:
            run_copy<nntile::bool_t>(graph, attrs, x_name, y_name);
            break;
        default:
            throw std::runtime_error("Unsupported data type for copy");
    }
}

//! Execute transpose operation
void execute_transpose(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const TransposeAttrs& attrs = std::get<TransposeAttrs>(op_info.attrs);
    const std::string& x_name = op_info.input_names[0];
    const std::string& y_name = op_info.output_names[0];
    DataType dtype = graph.get_dtype(x_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_transpose<nntile::fp32_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_transpose<nntile::fp32_fast_tf32_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_transpose<nntile::fp32_fast_fp16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_transpose<nntile::fp32_fast_bf16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP64:
            run_transpose<nntile::fp64_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP16:
            run_transpose<nntile::fp16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::BF16:
            run_transpose<nntile::bf16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for transpose operation");
        default:
            throw std::runtime_error("Unsupported data type for transpose");
    }
}

//! Execute gather operation
void execute_gather(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const ClearAttrs& attrs = std::get<ClearAttrs>(op_info.attrs);
    const std::string& x_name = op_info.input_names[0];
    const std::string& y_name = op_info.output_names[0];
    DataType dtype = graph.get_dtype(x_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_gather<nntile::fp32_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_gather<nntile::fp32_fast_tf32_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_gather<nntile::fp32_fast_fp16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_gather<nntile::fp32_fast_bf16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP64:
            run_gather<nntile::fp64_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP16:
            run_gather<nntile::fp16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::BF16:
            run_gather<nntile::bf16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::INT64:
            run_gather<nntile::int64_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::INT32:
            run_gather<nntile::int32_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::BOOL:
            run_gather<nntile::bool_t>(graph, attrs, x_name, y_name);
            break;
        default:
            throw std::runtime_error("Unsupported data type for gather");
    }
}

//! Execute hypot_scalar_inverse operation
void execute_hypot_scalar_inverse(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const HypotScalarInverseAttrs& attrs = std::get<HypotScalarInverseAttrs>(op_info.attrs);
    const std::string& x_name = op_info.input_names[0];
    DataType dtype = graph.get_dtype(x_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_hypot_scalar_inverse<nntile::fp32_t>(graph, attrs, x_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_hypot_scalar_inverse<nntile::fp32_fast_tf32_t>(graph, attrs, x_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_hypot_scalar_inverse<nntile::fp32_fast_fp16_t>(graph, attrs, x_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_hypot_scalar_inverse<nntile::fp32_fast_bf16_t>(graph, attrs, x_name);
            break;
        case DataType::FP64:
            run_hypot_scalar_inverse<nntile::fp64_t>(graph, attrs, x_name);
            break;
        case DataType::FP16:
            run_hypot_scalar_inverse<nntile::fp16_t>(graph, attrs, x_name);
            break;
        case DataType::BF16:
            run_hypot_scalar_inverse<nntile::bf16_t>(graph, attrs, x_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for hypot_scalar_inverse operation");
        default:
            throw std::runtime_error("Unsupported data type for hypot_scalar_inverse");
    }
}

//! Execute subtract_indexed_outputs operation
void execute_subtract_indexed_outputs(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const SubtractIndexedOutputsAttrs& attrs = std::get<SubtractIndexedOutputsAttrs>(op_info.attrs);
    const std::string& labels_name = op_info.input_names[0];
    const std::string& x_name = op_info.input_names[1];  // x is both input and output
    DataType dtype = graph.get_dtype(x_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_subtract_indexed_outputs<nntile::fp32_t>(graph, attrs, labels_name, x_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_subtract_indexed_outputs<nntile::fp32_fast_tf32_t>(graph, attrs, labels_name, x_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_subtract_indexed_outputs<nntile::fp32_fast_fp16_t>(graph, attrs, labels_name, x_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_subtract_indexed_outputs<nntile::fp32_fast_bf16_t>(graph, attrs, labels_name, x_name);
            break;
        case DataType::FP64:
            run_subtract_indexed_outputs<nntile::fp64_t>(graph, attrs, labels_name, x_name);
            break;
        case DataType::FP16:
            run_subtract_indexed_outputs<nntile::fp16_t>(graph, attrs, labels_name, x_name);
            break;
        case DataType::BF16:
            run_subtract_indexed_outputs<nntile::bf16_t>(graph, attrs, labels_name, x_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for subtract_indexed_outputs operation");
        default:
            throw std::runtime_error("Unsupported data type for subtract_indexed_outputs");
    }
}

} // namespace nntile::graph
