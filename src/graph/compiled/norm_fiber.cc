/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/compiled/norm_fiber.cc
 * Compiled graph norm_fiber operation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/compiled/norm_fiber.hh"
#include "nntile/graph/logical/norm_fiber.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/tensor/norm_fiber.hh"

namespace nntile::graph
{

namespace
{

template<typename T>
void run_norm_fiber(CompiledGraph& graph, const ReductionAttrs& attrs,
                    const std::string& x_name, const std::string& src2_name,
                    const std::string& dst_name)
{
    auto& x = graph.get_tensor<T>(x_name);
    auto& src2 = graph.get_tensor<T>(src2_name);
    auto& dst = graph.get_tensor<T>(dst_name);

    const auto alpha = static_cast<nntile::Scalar>(attrs.alpha);
    const auto beta = static_cast<nntile::Scalar>(attrs.beta);

    nntile::tensor::norm_fiber<T>(alpha, x, beta, src2, dst, attrs.axis, attrs.batch_ndim, attrs.redux);
}

} // namespace

void execute_norm_fiber(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const ReductionAttrs& attrs = *std::static_pointer_cast<ReductionAttrs>(op_info.attrs);
    const std::string& x_name = op_info.input_names[0];
    const std::string& src2_name = op_info.input_names[1];
    const std::string& dst_name = op_info.output_names[0];
    DataType dtype = graph.get_dtype(x_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_norm_fiber<nntile::fp32_t>(graph, attrs, x_name, src2_name, dst_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_norm_fiber<nntile::fp32_fast_tf32_t>(graph, attrs, x_name, src2_name, dst_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_norm_fiber<nntile::fp32_fast_fp16_t>(graph, attrs, x_name, src2_name, dst_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_norm_fiber<nntile::fp32_fast_bf16_t>(graph, attrs, x_name, src2_name, dst_name);
            break;
        case DataType::FP64:
            run_norm_fiber<nntile::fp64_t>(graph, attrs, x_name, src2_name, dst_name);
            break;
        case DataType::FP16:
            throw std::runtime_error("FP16 data type not supported for norm_fiber operation");
            break;
        case DataType::BF16:
            run_norm_fiber<nntile::bf16_t>(graph, attrs, x_name, src2_name, dst_name);
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

} // namespace nntile::graph
