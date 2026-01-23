/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/compiled/gelu_backward.cc
 * Compiled graph gelu_backward operation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/compiled/gelu_backward.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/tensor/gelu_backward.hh"

namespace nntile::graph
{

namespace
{

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

} // namespace

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

} // namespace nntile::graph
