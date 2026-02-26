/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/compiled/copy_intersection.cc
 * Compiled graph copy_intersection operation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/compiled/copy_intersection.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/tensor/copy_intersection.hh"

namespace nntile::graph
{

namespace
{

template<typename T>
void run_copy_intersection(CompiledGraph& graph, const CopyIntersectionAttrs& attrs,
                          const std::string& src_name, const std::string& dst_name)
{
    auto& src = graph.get_tensor<T>(src_name);
    auto& dst = graph.get_tensor<T>(dst_name);

    nntile::tensor::copy_intersection<T>(src, attrs.src_offset, dst, attrs.dst_offset);
}

} // namespace

void execute_copy_intersection(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const CopyIntersectionAttrs& attrs = std::get<CopyIntersectionAttrs>(op_info.attrs);
    const std::string& src_name = op_info.input_names[0];
    const std::string& dst_name = op_info.output_names[0];
    DataType dtype = graph.get_dtype(src_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_copy_intersection<nntile::fp32_t>(graph, attrs, src_name, dst_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_copy_intersection<nntile::fp32_fast_tf32_t>(graph, attrs, src_name, dst_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_copy_intersection<nntile::fp32_fast_fp16_t>(graph, attrs, src_name, dst_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_copy_intersection<nntile::fp32_fast_bf16_t>(graph, attrs, src_name, dst_name);
            break;
        case DataType::FP64:
            run_copy_intersection<nntile::fp64_t>(graph, attrs, src_name, dst_name);
            break;
        case DataType::FP16:
            run_copy_intersection<nntile::fp16_t>(graph, attrs, src_name, dst_name);
            break;
        case DataType::BF16:
            run_copy_intersection<nntile::bf16_t>(graph, attrs, src_name, dst_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for copy_intersection operation");
        default:
            throw std::runtime_error("Unsupported data type for copy_intersection");
    }
}

} // namespace nntile::graph
