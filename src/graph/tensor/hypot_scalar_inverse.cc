/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/hypot_scalar_inverse.cc
 * TensorGraph hypot_scalar_inverse operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/hypot_scalar_inverse.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/hypot_scalar_inverse.hh"

namespace nntile::graph
{

namespace
{

template<typename T>
void run_hypot_scalar_inverse(
    TensorGraph::ExecutionContext& ctx,
    Scalar eps, Scalar alpha,
    TensorGraph::TensorNode* dst)
{
    auto& dst_t = ctx.get_tensor<T>(dst);
    nntile::tensor::hypot_scalar_inverse<T>(eps, alpha, dst_t);
}

} // namespace

void hypot_scalar_inverse(
    Scalar eps,
    Scalar alpha,
    TensorGraph::TensorNode* dst)
{
    if(dst == nullptr)
    {
        throw std::invalid_argument(
            "hypot_scalar_inverse: dst tensor must be non-null");
    }

    auto op = std::make_shared<TensorHypotScalarInverseOp>(eps, alpha, dst);
    dst->graph()->add_op(op);
}

void TensorHypotScalarInverseOp::execute(
    TensorGraph::ExecutionContext& ctx) const
{
    DataType dtype = ctx.get_dtype(dst);

    switch(dtype)
    {
        case DataType::FP32:
            run_hypot_scalar_inverse<nntile::fp32_t>(ctx, eps, alpha, dst);
            break;
        case DataType::FP32_FAST_TF32:
            run_hypot_scalar_inverse<nntile::fp32_fast_tf32_t>(ctx, eps, alpha, dst);
            break;
        case DataType::FP32_FAST_FP16:
            run_hypot_scalar_inverse<nntile::fp32_fast_fp16_t>(ctx, eps, alpha, dst);
            break;
        case DataType::FP32_FAST_BF16:
            run_hypot_scalar_inverse<nntile::fp32_fast_bf16_t>(ctx, eps, alpha, dst);
            break;
        case DataType::FP64:
            run_hypot_scalar_inverse<nntile::fp64_t>(ctx, eps, alpha, dst);
            break;
        case DataType::FP16:
            run_hypot_scalar_inverse<nntile::fp16_t>(ctx, eps, alpha, dst);
            break;
        case DataType::BF16:
            run_hypot_scalar_inverse<nntile::bf16_t>(ctx, eps, alpha, dst);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for hypot_scalar_inverse operation");
        default:
            throw std::runtime_error(
                "Unsupported data type for hypot_scalar_inverse");
    }
}

} // namespace nntile::graph
