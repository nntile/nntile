/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/compiled/total_sum_accum.cc
 * Compiled graph total_sum_accum operation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/compiled/total_sum_accum.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/tensor/total_sum_accum.hh"

namespace nntile::graph
{

namespace
{

template<typename T>
void run_total_sum_accum(CompiledGraph& graph, const TotalSumAccumAttrs& attrs,
                       const std::string& logsumexp_name, const std::string& src_name,
                       const std::string& class_labels_name, const std::string& val_name)
{
    auto& logsumexp = graph.get_tensor<T>(logsumexp_name);
    auto& src = graph.get_tensor<T>(src_name);
    auto& class_labels = graph.get_tensor<nntile::int64_t>(class_labels_name);
    auto& val = graph.get_tensor<nntile::fp32_t>(val_name);

    const auto alpha = static_cast<nntile::Scalar>(attrs.alpha);

    nntile::tensor::total_sum_accum<T>(alpha, logsumexp, src, class_labels, val, attrs.ignore_index);
}

} // namespace

//! Execute total_sum_accum operation
void execute_total_sum_accum(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const TotalSumAccumAttrs& attrs = std::get<TotalSumAccumAttrs>(op_info.attrs);
    const std::string& logsumexp_name = op_info.input_names[0];
    const std::string& src_name = op_info.input_names[1];
    const std::string& class_labels_name = op_info.input_names[2];
    const std::string& val_name = op_info.output_names[0];
    DataType dtype = graph.get_dtype(src_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_total_sum_accum<nntile::fp32_t>(graph, attrs, logsumexp_name, src_name, class_labels_name, val_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_total_sum_accum<nntile::fp32_fast_tf32_t>(graph, attrs, logsumexp_name, src_name, class_labels_name, val_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_total_sum_accum<nntile::fp32_fast_fp16_t>(graph, attrs, logsumexp_name, src_name, class_labels_name, val_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_total_sum_accum<nntile::fp32_fast_bf16_t>(graph, attrs, logsumexp_name, src_name, class_labels_name, val_name);
            break;
        case DataType::FP64:
            run_total_sum_accum<nntile::fp64_t>(graph, attrs, logsumexp_name, src_name, class_labels_name, val_name);
            break;
        case DataType::FP16:
            run_total_sum_accum<nntile::fp16_t>(graph, attrs, logsumexp_name, src_name, class_labels_name, val_name);
            break;
        case DataType::BF16:
            run_total_sum_accum<nntile::bf16_t>(graph, attrs, logsumexp_name, src_name, class_labels_name, val_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for total_sum_accum operation");
        default:
            throw std::runtime_error("Unsupported data type for total_sum_accum");
    }
}

} // namespace nntile::graph