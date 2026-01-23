/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/compiled/add_fiber_inplace.cc
 * Compiled graph add_fiber_inplace operation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/compiled/add_fiber_inplace.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/tensor/add_fiber_inplace.hh"

namespace nntile::graph
{

namespace
{

template<typename T>
void run_add_fiber_inplace(CompiledGraph& graph, const ReductionAttrs& attrs,
                         const std::string& x_name, const std::string& y_name)
{
    auto& x = graph.get_tensor<T>(x_name);
    auto& y = graph.get_tensor<T>(y_name);

    const auto alpha = static_cast<nntile::Scalar>(attrs.alpha);
    const auto beta = static_cast<nntile::Scalar>(attrs.beta);

    nntile::tensor::add_fiber_inplace<T>(alpha, x, beta, y, attrs.axis, attrs.batch_ndim);
}

} // namespace

//! Execute add_fiber_inplace operation
void execute_add_fiber_inplace(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const ReductionAttrs& attrs = std::get<ReductionAttrs>(op_info.attrs);
    const std::string& x_name = op_info.input_names[0];
    const std::string& y_name = op_info.output_names[0];
    DataType dtype = graph.get_dtype(x_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_add_fiber_inplace<nntile::fp32_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_add_fiber_inplace<nntile::fp32_fast_tf32_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_add_fiber_inplace<nntile::fp32_fast_fp16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_add_fiber_inplace<nntile::fp32_fast_bf16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP64:
            run_add_fiber_inplace<nntile::fp64_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::FP16:
            run_add_fiber_inplace<nntile::fp16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::BF16:
            run_add_fiber_inplace<nntile::bf16_t>(graph, attrs, x_name, y_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for add_fiber_inplace operation");
        default:
            throw std::runtime_error("Unsupported data type for add_fiber_inplace");
    }
}

} // namespace nntile::graph