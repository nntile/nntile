/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/sumprod_fiber.cc
 * TensorGraph sumprod_fiber operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/sumprod_fiber.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/sumprod_fiber.hh"

namespace nntile::graph::tensor
{

namespace
{

template<typename T>
void run_sumprod_fiber(
    TensorGraph::Runtime& runtime,
    Scalar alpha, Scalar beta,
    Index axis, int redux,
    TensorGraph::TensorNode* src1,
    TensorGraph::TensorNode* src2,
    TensorGraph::TensorNode* dst)
{
    auto& src1_t = runtime.get_tensor<T>(src1);
    auto& src2_t = runtime.get_tensor<T>(src2);
    auto& dst_t = runtime.get_tensor<T>(dst);
    nntile::tensor::sumprod_fiber<T>(
        alpha, src1_t, src2_t, beta, dst_t, axis, redux);
}

} // namespace

void sumprod_fiber(
    TensorGraph::TensorNode* src1,
    TensorGraph::TensorNode* src2,
    TensorGraph::TensorNode* dst,
    Index axis,
    int redux,
    Scalar alpha,
    Scalar beta)
{
    if(src1 == nullptr || src2 == nullptr || dst == nullptr)
    {
        throw std::invalid_argument(
            "sumprod_fiber: input tensors must be non-null");
    }
    if(src1->graph() != src2->graph() || src1->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "sumprod_fiber: input tensors must belong to the same graph");
    }
    if(src1->dtype() != src2->dtype() || src1->dtype() != dst->dtype())
    {
        throw std::invalid_argument(
            "sumprod_fiber: input tensors must have the same dtype");
    }
    if(src1->shape() != src2->shape())
    {
        throw std::invalid_argument(
            "sumprod_fiber: src1 and src2 must have the same shape");
    }
    if(dst->ndim() != 1)
    {
        throw std::invalid_argument(
            "sumprod_fiber: dst must be 1-dimensional");
    }
    if(axis < 0 || axis >= src1->ndim())
    {
        throw std::invalid_argument(
            "sumprod_fiber: axis out of range");
    }
    if(dst->dim(0) != src1->dim(axis))
    {
        throw std::invalid_argument(
            "sumprod_fiber: dst.shape[0] must equal src1.shape[axis]");
    }
    if(src1 == src2 || src1 == dst || src2 == dst)
    {
        throw std::invalid_argument(
            "sumprod_fiber: src1, src2, and dst must be distinct tensors");
    }

    auto op = std::make_shared<TensorSumprodFiberOp>(
        src1, src2, dst, axis, redux, alpha, beta);
    src1->graph()->add_op(op);
}

void TensorSumprodFiberOp::execute(
    TensorGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(src1);

    switch(dtype)
    {
        case DataType::FP32:
            run_sumprod_fiber<nntile::fp32_t>(
                runtime, alpha, beta, axis, redux, src1, src2, dst);
            break;
        case DataType::FP32_FAST_TF32:
            run_sumprod_fiber<nntile::fp32_fast_tf32_t>(
                runtime, alpha, beta, axis, redux, src1, src2, dst);
            break;
        case DataType::FP32_FAST_FP16:
            run_sumprod_fiber<nntile::fp32_fast_fp16_t>(
                runtime, alpha, beta, axis, redux, src1, src2, dst);
            break;
        case DataType::FP32_FAST_BF16:
            run_sumprod_fiber<nntile::fp32_fast_bf16_t>(
                runtime, alpha, beta, axis, redux, src1, src2, dst);
            break;
        case DataType::FP64:
            run_sumprod_fiber<nntile::fp64_t>(
                runtime, alpha, beta, axis, redux, src1, src2, dst);
            break;
        case DataType::FP16:
            run_sumprod_fiber<nntile::fp16_t>(
                runtime, alpha, beta, axis, redux, src1, src2, dst);
            break;
        case DataType::BF16:
            run_sumprod_fiber<nntile::bf16_t>(
                runtime, alpha, beta, axis, redux, src1, src2, dst);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for sumprod_fiber operation");
        default:
            throw std::runtime_error("Unsupported data type for sumprod_fiber");
    }
}

} // namespace nntile::graph::tensor
