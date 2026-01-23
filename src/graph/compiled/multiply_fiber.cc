/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/compiled/multiply_fiber.cc
 * Compiled graph multiply_fiber operation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/compiled/multiply_fiber.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/tensor/multiply_fiber.hh"

namespace nntile::graph
{

namespace
{

template<typename T>
void run_multiply_fiber(CompiledGraph& graph, const ReductionAttrs& attrs,
                      const std::string& x_name, const std::string& y_name,
                      const std::string& z_name)
{
    auto& x = graph.get_tensor<T>(x_name);
    auto& y = graph.get_tensor<T>(y_name);
    auto& z = graph.get_tensor<T>(z_name);

    const auto alpha = static_cast<nntile::Scalar>(attrs.alpha);

    nntile::tensor::multiply_fiber<T>(alpha, x, y, z, attrs.axis);
}

} // namespace

//! Execute multiply_fiber operation
void execute_multiply_fiber(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const ReductionAttrs& attrs = std::get<ReductionAttrs>(op_info.attrs);
    const std::string& x_name = op_info.input_names[0];
    const std::string& y_name = op_info.input_names[1];
    const std::string& z_name = op_info.output_names[0];
    DataType dtype = graph.get_dtype(x_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_multiply_fiber<nntile::fp32_t>(graph, attrs, x_name, y_name, z_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_multiply_fiber<nntile::fp32_fast_tf32_t>(graph, attrs, x_name, y_name, z_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_multiply_fiber<nntile::fp32_fast_fp16_t>(graph, attrs, x_name, y_name, z_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_multiply_fiber<nntile::fp32_fast_bf16_t>(graph, attrs, x_name, y_name, z_name);
            break;
        case DataType::FP64:
            run_multiply_fiber<nntile::fp64_t>(graph, attrs, x_name, y_name, z_name);
            break;
        case DataType::FP16:
            run_multiply_fiber<nntile::fp16_t>(graph, attrs, x_name, y_name, z_name);
            break;
        case DataType::BF16:
            run_multiply_fiber<nntile::bf16_t>(graph, attrs, x_name, y_name, z_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for multiply_fiber operation");
        default:
            throw std::runtime_error("Unsupported data type for multiply_fiber");
    }
}

} // namespace nntile::graph