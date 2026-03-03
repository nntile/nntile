/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/log_scalar.cc
 * TensorGraph log_scalar operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/log_scalar.hh"

#include <stdexcept>

#include "nntile/graph/tensor.hh"
#include "nntile/tensor/log_scalar.hh"

namespace nntile::graph
{

namespace
{

template<typename T>
void run_log_scalar(TensorGraph::ExecutionContext& ctx,
                   const std::string& name,
                   TensorGraph::TensorNode* value)
{
    auto& value_t = ctx.get_tensor<T>(value);
    nntile::tensor::log_scalar<T>(name, value_t);
}

} // namespace

void log_scalar(const std::string& name,
                TensorGraph::TensorNode* value)
{
    if(value == nullptr)
        throw std::invalid_argument("log_scalar: value tensor must be non-null");
    auto op = std::make_shared<TensorLogScalarOp>(name, value);
    value->graph()->add_op(op);
}

void TensorLogScalarOp::execute(TensorGraph::ExecutionContext& ctx) const
{
    DataType dtype = ctx.get_dtype(value);
    switch(dtype)
    {
        case DataType::FP32:
            run_log_scalar<nntile::fp32_t>(ctx, name, value);
            break;
        case DataType::FP32_FAST_TF32:
            run_log_scalar<nntile::fp32_fast_tf32_t>(ctx, name, value);
            break;
        case DataType::FP32_FAST_FP16:
            run_log_scalar<nntile::fp32_fast_fp16_t>(ctx, name, value);
            break;
        case DataType::FP32_FAST_BF16:
            run_log_scalar<nntile::fp32_fast_bf16_t>(ctx, name, value);
            break;
        case DataType::FP64:
            run_log_scalar<nntile::fp64_t>(ctx, name, value);
            break;
        case DataType::FP16:
            throw std::runtime_error(
                "FP16 not supported for log_scalar (use FP32_FAST_FP16)");
            break;
        case DataType::BF16:
            run_log_scalar<nntile::bf16_t>(ctx, name, value);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " not supported for log_scalar");
        default:
            throw std::runtime_error("Unsupported data type for log_scalar");
    }
}

} // namespace nntile::graph
