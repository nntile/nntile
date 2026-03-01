/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/compiled/sum.cc
 * Compiled graph sum operation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/compiled/sum.hh"
#include "nntile/graph/logical/norm.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/tensor/sum.hh"

namespace nntile::graph
{

namespace
{

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

} // namespace

//! Execute sum operation
void execute_sum(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const TotalSumAttrs& attrs = *std::static_pointer_cast<TotalSumAttrs>(op_info.attrs);
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

} // namespace nntile::graph
