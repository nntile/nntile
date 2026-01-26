/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/compiled/fill.cc
 * Compiled graph fill operation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/compiled/fill.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/tensor/fill.hh"

namespace nntile::graph
{

namespace
{

// Utility operations
template<typename T>
void run_fill(CompiledGraph& graph, const FillAttrs& attrs,
              const std::string& x_name)
{
    auto& x = graph.get_tensor<T>(x_name);

    nntile::tensor::fill<T>(attrs.val, x);
}

} // namespace

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
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for fill operation");
        default:
            throw std::runtime_error("Unsupported data type for fill");
    }
}

} // namespace nntile::graph
