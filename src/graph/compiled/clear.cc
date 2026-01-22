/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/compiled/clear.cc
 * CLEAR operation execution for CompiledGraph.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/compiled/clear.hh"

// Include standard headers
#include <stdexcept>

// Include other NNTile headers
#include "nntile/base_types.hh"
#include "nntile/tensor/clear.hh"

namespace nntile::graph
{

namespace
{

template<typename T>
void run_clear(CompiledGraph& graph, const std::string& name)
{
    auto& tensor = graph.get_tensor<T>(name);
    nntile::tensor::clear<T>(tensor);
}

} // namespace

//! Execute clear operation
void execute_clear(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const std::string& output_name = op_info.output_names[0];
    DataType dtype = graph.get_dtype(output_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_clear<nntile::fp32_t>(graph, output_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_clear<nntile::fp32_fast_tf32_t>(graph, output_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_clear<nntile::fp32_fast_fp16_t>(graph, output_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_clear<nntile::fp32_fast_bf16_t>(graph, output_name);
            break;
        case DataType::FP64:
            run_clear<nntile::fp64_t>(graph, output_name);
            break;
        case DataType::FP16:
            run_clear<nntile::fp16_t>(graph, output_name);
            break;
        case DataType::BF16:
            run_clear<nntile::bf16_t>(graph, output_name);
            break;
        case DataType::INT64:
            run_clear<nntile::int64_t>(graph, output_name);
            break;
        case DataType::BOOL:
            run_clear<nntile::bool_t>(graph, output_name);
            break;
        case DataType::INT32:
            throw std::runtime_error(
                "INT32 data type not supported for clear operation");
        default:
            throw std::runtime_error("Unsupported data type for clear");
    }
}

} // namespace nntile::graph
