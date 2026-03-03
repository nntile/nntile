/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/silu_inplace.cc
 * TensorGraph silu_inplace operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/silu_inplace.hh"

#include <stdexcept>

#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/silu_inplace.hh"

namespace nntile::graph
{

namespace
{

template<typename T>
void run_silu_inplace(
    TensorGraph::ExecutionContext& ctx,
    TensorGraph::TensorNode* dst)
{
    auto& dst_t = ctx.get_tensor<T>(dst);
    nntile::tensor::silu_inplace<T>(dst_t);
}

} // namespace

void silu_inplace(TensorGraph::TensorNode* dst)
{
    if(dst == nullptr)
    {
        throw std::invalid_argument(
            "silu_inplace: dst tensor must be non-null");
    }

    auto op = std::make_shared<TensorSiluInplaceOp>(dst);
    dst->graph()->add_op(op);
}

void TensorSiluInplaceOp::execute(
    TensorGraph::ExecutionContext& ctx) const
{
    DataType dtype = ctx.get_dtype(dst);

    switch(dtype)
    {
        case DataType::FP32:
            run_silu_inplace<nntile::fp32_t>(ctx, dst);
            break;
        case DataType::FP32_FAST_TF32:
            run_silu_inplace<nntile::fp32_fast_tf32_t>(ctx, dst);
            break;
        case DataType::FP32_FAST_FP16:
            run_silu_inplace<nntile::fp32_fast_fp16_t>(ctx, dst);
            break;
        case DataType::FP32_FAST_BF16:
            run_silu_inplace<nntile::fp32_fast_bf16_t>(ctx, dst);
            break;
        case DataType::FP64:
            run_silu_inplace<nntile::fp64_t>(ctx, dst);
            break;
        case DataType::FP16:
            run_silu_inplace<nntile::fp16_t>(ctx, dst);
            break;
        case DataType::BF16:
            run_silu_inplace<nntile::bf16_t>(ctx, dst);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for silu_inplace operation");
        default:
            throw std::runtime_error("Unsupported data type for silu_inplace");
    }
}

} // namespace nntile::graph
