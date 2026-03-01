/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/compiled/multiply_slice.cc
 * Compiled graph multiply_slice operation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/compiled/multiply_slice.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/tensor/multiply_slice.hh"

namespace nntile::graph
{

namespace
{

template<typename T>
void run_multiply_slice(CompiledGraph& graph, const MultiplySliceAttrs& attrs,
                      const std::string& slice_name, const std::string& tensor_name)
{
    auto& slice = graph.get_tensor<T>(slice_name);
    auto& tensor = graph.get_tensor<T>(tensor_name);

    const auto alpha = static_cast<nntile::Scalar>(attrs.alpha);

    nntile::tensor::multiply_slice<T>(alpha, slice, tensor, attrs.axis);
}

} // namespace

//! Execute multiply_slice operation
void execute_multiply_slice(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const MultiplySliceAttrs& attrs = *std::static_pointer_cast<MultiplySliceAttrs>(op_info.attrs);
    const std::string& slice_name = op_info.input_names[0];
    const std::string& tensor_name = op_info.output_names[0];
    DataType dtype = graph.get_dtype(slice_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_multiply_slice<nntile::fp32_t>(graph, attrs, slice_name, tensor_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_multiply_slice<nntile::fp32_fast_tf32_t>(graph, attrs, slice_name, tensor_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_multiply_slice<nntile::fp32_fast_fp16_t>(graph, attrs, slice_name, tensor_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_multiply_slice<nntile::fp32_fast_bf16_t>(graph, attrs, slice_name, tensor_name);
            break;
        case DataType::FP64:
            run_multiply_slice<nntile::fp64_t>(graph, attrs, slice_name, tensor_name);
            break;
        case DataType::FP16:
            run_multiply_slice<nntile::fp16_t>(graph, attrs, slice_name, tensor_name);
            break;
        case DataType::BF16:
            run_multiply_slice<nntile::bf16_t>(graph, attrs, slice_name, tensor_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for multiply_slice operation");
        default:
            throw std::runtime_error("Unsupported data type for multiply_slice");
    }
}

} // namespace nntile::graph
