/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/compiled/gemm.cc
 * Compiled graph gemm operation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/compiled/gemm.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/tensor/gemm.hh"

namespace nntile::graph
{

namespace
{

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

} // namespace

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

} // namespace nntile::graph
