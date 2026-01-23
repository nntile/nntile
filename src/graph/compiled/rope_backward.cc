/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/compiled/rope_backward.cc
 * Compiled graph rope_backward operation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/compiled/rope_backward.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/tensor/rope_backward.hh"

namespace nntile::graph
{

namespace
{

template<typename T>
void run_rope_backward(CompiledGraph& graph, const ClearAttrs& attrs,
                       const std::string& sin_name, const std::string& cos_name,
                       const std::string& dy_name, const std::string& dx_name)
{
    auto& sin_tensor = graph.get_tensor<T>(sin_name);
    auto& cos_tensor = graph.get_tensor<T>(cos_name);
    auto& dy = graph.get_tensor<T>(dy_name);
    auto& dx = graph.get_tensor<T>(dx_name);

    nntile::tensor::rope_backward<T>(sin_tensor, cos_tensor, dy, dx);
}

} // namespace

void execute_rope_backward(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const ClearAttrs& attrs = std::get<ClearAttrs>(op_info.attrs);
    const std::string& sin_name = op_info.input_names[0];
    const std::string& cos_name = op_info.input_names[1];
    const std::string& dy_name = op_info.input_names[2];
    const std::string& dx_name = op_info.output_names[0];
    DataType dtype = graph.get_dtype(sin_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_rope_backward<nntile::fp32_t>(graph, attrs, sin_name, cos_name, dy_name, dx_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_rope_backward<nntile::fp32_fast_tf32_t>(graph, attrs, sin_name, cos_name, dy_name, dx_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_rope_backward<nntile::fp32_fast_fp16_t>(graph, attrs, sin_name, cos_name, dy_name, dx_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_rope_backward<nntile::fp32_fast_bf16_t>(graph, attrs, sin_name, cos_name, dy_name, dx_name);
            break;
        case DataType::FP64:
            run_rope_backward<nntile::fp64_t>(graph, attrs, sin_name, cos_name, dy_name, dx_name);
            break;
        case DataType::FP16:
            run_rope_backward<nntile::fp16_t>(graph, attrs, sin_name, cos_name, dy_name, dx_name);
            break;
        case DataType::BF16:
            run_rope_backward<nntile::bf16_t>(graph, attrs, sin_name, cos_name, dy_name, dx_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for rope_backward operation");
        default:
            throw std::runtime_error("Unsupported data type for rope_backward");
    }
}

} // namespace nntile::graph
