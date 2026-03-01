/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/compiled/hypot_scalar_inverse.cc
 * Compiled graph hypot_scalar_inverse operation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/compiled/hypot_scalar_inverse.hh"
#include "nntile/graph/logical/hypot_scalar_inverse.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/tensor/hypot_scalar_inverse.hh"

namespace nntile::graph
{

namespace
{

// Element-wise operations
template<typename T>
void run_hypot_scalar_inverse(CompiledGraph& graph, const HypotScalarInverseAttrs& attrs,
                              const std::string& x_name)
{
    auto& x = graph.get_tensor<T>(x_name);

    nntile::tensor::hypot_scalar_inverse<T>(attrs.eps, attrs.alpha, x);
}

} // namespace

void execute_hypot_scalar_inverse(CompiledGraph& graph, const OpExecutionInfo& op_info)
{
    const HypotScalarInverseAttrs& attrs = *std::static_pointer_cast<HypotScalarInverseAttrs>(op_info.attrs);
    const std::string& x_name = op_info.input_names[0];
    DataType dtype = graph.get_dtype(x_name);

    switch(dtype)
    {
        case DataType::FP32:
            run_hypot_scalar_inverse<nntile::fp32_t>(graph, attrs, x_name);
            break;
        case DataType::FP32_FAST_TF32:
            run_hypot_scalar_inverse<nntile::fp32_fast_tf32_t>(graph, attrs, x_name);
            break;
        case DataType::FP32_FAST_FP16:
            run_hypot_scalar_inverse<nntile::fp32_fast_fp16_t>(graph, attrs, x_name);
            break;
        case DataType::FP32_FAST_BF16:
            run_hypot_scalar_inverse<nntile::fp32_fast_bf16_t>(graph, attrs, x_name);
            break;
        case DataType::FP64:
            run_hypot_scalar_inverse<nntile::fp64_t>(graph, attrs, x_name);
            break;
        case DataType::FP16:
            run_hypot_scalar_inverse<nntile::fp16_t>(graph, attrs, x_name);
            break;
        case DataType::BF16:
            run_hypot_scalar_inverse<nntile::bf16_t>(graph, attrs, x_name);
            break;
        case DataType::INT64:
        case DataType::INT32:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for hypot_scalar_inverse operation");
        default:
            throw std::runtime_error("Unsupported data type for hypot_scalar_inverse");
    }
}

} // namespace nntile::graph
