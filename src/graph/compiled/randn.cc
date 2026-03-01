/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/compiled/randn.cc
 * Compiled graph randn operation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/compiled/randn.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/tensor/randn.hh"

namespace nntile::graph
{

namespace
{

template<typename T>
void run_randn(CompiledGraph& graph, const RandnAttrs& attrs,
               const std::string& x_name)
{
    auto& x = graph.get_tensor<T>(x_name);

    nntile::tensor::randn<T>(x, attrs.start, attrs.underlying_shape,
                            attrs.seed, attrs.mean, attrs.stddev);
}

} // namespace

void execute_randn(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const RandnAttrs& attrs = *std::static_pointer_cast<RandnAttrs>(op_info.attrs);
    const std::string& x_name = op_info.output_names[0];
    DataType dtype = graph.get_dtype(x_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_randn<nntile::fp32_t>(graph, attrs, x_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_randn<nntile::fp32_fast_tf32_t>(graph, attrs, x_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_randn<nntile::fp32_fast_fp16_t>(graph, attrs, x_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_randn<nntile::fp32_fast_bf16_t>(graph, attrs, x_name);
            break;
        case DataType::FP64:
            run_randn<nntile::fp64_t>(graph, attrs, x_name);
            break;
        case DataType::FP16:
            throw std::runtime_error("FP16 data type not supported for randn operation");
            break;
        case DataType::BF16:
            run_randn<nntile::bf16_t>(graph, attrs, x_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for randn operation");
        default:
            throw std::runtime_error("Unsupported data type for randn");
    }
}

} // namespace nntile::graph
