/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/compiled/sumprod_slice.cc
 * Compiled graph sumprod_slice operation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/compiled/sumprod_slice.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/tensor/sumprod_slice.hh"

namespace nntile::graph
{

namespace
{

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

} // namespace

void execute_sumprod_slice(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const ReductionAttrs& attrs = *std::static_pointer_cast<ReductionAttrs>(op_info.attrs);
    const std::string& x1_name = op_info.input_names[0];
    const std::string& x2_name = op_info.input_names[1];
    const std::string& y_name = op_info.output_names[0];
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

} // namespace nntile::graph
