/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/add_scalar_scaled_inplace.cc
 * TensorGraph add_scalar_scaled_inplace implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/add_scalar_scaled_inplace.hh"

#include <starpu.h>

#include "nntile/base_types.hh"
#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/add_inplace.hh"

namespace nntile::graph::tensor
{

namespace
{

template<typename T>
void run_add_scalar_scaled_inplace(
    TensorGraph::Runtime& runtime,
    Scalar alpha, Scalar beta,
    TensorGraph::TensorNode* scalar_tensor,
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst)
{
    auto& scalar_t = runtime.get_tensor<T>(scalar_tensor);
    auto& src_t = runtime.get_tensor<T>(src);
    auto& dst_t = runtime.get_tensor<T>(dst);

    if(scalar_t.nelems != 1)
    {
        throw std::runtime_error(
            "add_scalar_scaled_inplace: scalar_tensor must have exactly 1 element");
    }

    Scalar scalar_val;
    {
        auto tile = scalar_t.get_tile(0);
        auto tile_local = tile.acquire(STARPU_R);
        // Use conversion operator (repr_t), not .value - fp16/bf16 store raw bits
        scalar_val = static_cast<Scalar>(static_cast<typename T::repr_t>(tile_local[0]));
        tile_local.release();
    }

    Scalar effective_alpha = alpha * scalar_val;
    nntile::tensor::add_inplace<T>(effective_alpha, src_t, beta, dst_t);
}

} // namespace

void add_scalar_scaled_inplace(
    Scalar alpha,
    TensorGraph::TensorNode* scalar_tensor,
    TensorGraph::TensorNode* src,
    Scalar beta,
    TensorGraph::TensorNode* dst)
{
    if(scalar_tensor == nullptr || src == nullptr || dst == nullptr)
    {
        throw std::invalid_argument(
            "add_scalar_scaled_inplace: input tensors must be non-null");
    }
    if(scalar_tensor->graph() != src->graph() || src->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "add_scalar_scaled_inplace: tensors must belong to the same graph");
    }
    if(scalar_tensor->dtype() != src->dtype() || src->dtype() != dst->dtype())
    {
        throw std::invalid_argument(
            "add_scalar_scaled_inplace: tensors must have the same dtype");
    }
    if(!scalar_tensor->shape().empty())
    {
        throw std::invalid_argument(
            "add_scalar_scaled_inplace: scalar_tensor must be 0-dimensional");
    }
    if(src->shape() != dst->shape())
    {
        throw std::invalid_argument(
            "add_scalar_scaled_inplace: src and dst must have the same shape");
    }
    if(src == dst)
    {
        throw std::invalid_argument(
            "add_scalar_scaled_inplace: src and dst must be distinct");
    }

    auto op = std::make_shared<TensorAddScalarScaledInplaceOp>(
        scalar_tensor, src, dst, alpha, beta);
    src->graph()->add_op(op);
}

void TensorAddScalarScaledInplaceOp::execute(
    TensorGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(src);

    switch(dtype)
    {
        case DataType::FP32:
            run_add_scalar_scaled_inplace<nntile::fp32_t>(
                runtime, alpha, beta, scalar_tensor, src, dst);
            break;
        case DataType::FP32_FAST_TF32:
            run_add_scalar_scaled_inplace<nntile::fp32_fast_tf32_t>(
                runtime, alpha, beta, scalar_tensor, src, dst);
            break;
        case DataType::FP32_FAST_FP16:
            run_add_scalar_scaled_inplace<nntile::fp32_fast_fp16_t>(
                runtime, alpha, beta, scalar_tensor, src, dst);
            break;
        case DataType::FP32_FAST_BF16:
            run_add_scalar_scaled_inplace<nntile::fp32_fast_bf16_t>(
                runtime, alpha, beta, scalar_tensor, src, dst);
            break;
        case DataType::FP64:
            run_add_scalar_scaled_inplace<nntile::fp64_t>(
                runtime, alpha, beta, scalar_tensor, src, dst);
            break;
        case DataType::FP16:
            run_add_scalar_scaled_inplace<nntile::fp16_t>(
                runtime, alpha, beta, scalar_tensor, src, dst);
            break;
        case DataType::BF16:
            run_add_scalar_scaled_inplace<nntile::bf16_t>(
                runtime, alpha, beta, scalar_tensor, src, dst);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for add_scalar_scaled_inplace");
        default:
            throw std::runtime_error(
                "Unsupported data type for add_scalar_scaled_inplace");
    }
}

} // namespace nntile::graph::tensor
