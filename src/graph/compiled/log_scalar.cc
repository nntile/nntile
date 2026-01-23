/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/compiled/log_scalar.cc
 * Compiled graph log_scalar operation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/compiled/log_scalar.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/tensor/log_scalar.hh"

namespace nntile::graph
{

namespace
{

template<typename T>
void run_log_scalar(CompiledGraph& graph, const LogScalarAttrs& attrs,
                    const std::string& x_name)
{
    auto& x = graph.get_tensor<T>(x_name);

    // Note: LogScalarAttrs.base is not used in the current tensor operation
    // The tensor::log_scalar takes a name parameter
    nntile::tensor::log_scalar<T>("tensor_value", x);
}

} // namespace

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
            throw std::runtime_error("FP16 data type not supported for log_scalar operation");
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

} // namespace nntile::graph
