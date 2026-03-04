/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/sum.cc
 * TensorGraph sum operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/sum.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/sum.hh"

namespace nntile::graph::tensor
{

namespace
{

template<typename T>
void run_sum(
    TensorGraph::Runtime& runtime,
    Scalar alpha, Scalar beta,
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst)
{
    auto& src_t = runtime.get_tensor<T>(src);
    auto& dst_t = runtime.get_tensor<T>(dst);
    nntile::tensor::sum<T>(alpha, src_t, beta, dst_t);
}

} // namespace

void sum(
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst,
    Scalar alpha,
    Scalar beta)
{
    if(src == nullptr || dst == nullptr)
    {
        throw std::invalid_argument(
            "sum: input tensors must be non-null");
    }
    if(src == dst)
    {
        throw std::invalid_argument(
            "sum: src and dst must be distinct tensors");
    }
    if(src->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "sum: input tensors must belong to the same graph");
    }
    if(src->dtype() != dst->dtype())
    {
        throw std::invalid_argument(
            "sum: input tensors must have the same dtype");
    }
    if(dst->ndim() != 0)
    {
        throw std::invalid_argument(
            "sum: dst must be a scalar (0-dimensional tensor)");
    }

    auto op = std::make_shared<TensorSumOp>(src, dst, alpha, beta);
    src->graph()->add_op(op);
}

void TensorSumOp::execute(
    TensorGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(src);

    switch(dtype)
    {
        case DataType::FP32:
            run_sum<nntile::fp32_t>(runtime, alpha, beta, src, dst);
            break;
        case DataType::FP32_FAST_TF32:
            run_sum<nntile::fp32_fast_tf32_t>(runtime, alpha, beta, src, dst);
            break;
        case DataType::FP32_FAST_FP16:
            run_sum<nntile::fp32_fast_fp16_t>(runtime, alpha, beta, src, dst);
            break;
        case DataType::FP32_FAST_BF16:
            run_sum<nntile::fp32_fast_bf16_t>(runtime, alpha, beta, src, dst);
            break;
        case DataType::FP64:
            run_sum<nntile::fp64_t>(runtime, alpha, beta, src, dst);
            break;
        case DataType::FP16:
            run_sum<nntile::fp16_t>(runtime, alpha, beta, src, dst);
            break;
        case DataType::BF16:
            run_sum<nntile::bf16_t>(runtime, alpha, beta, src, dst);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for sum operation");
        default:
            throw std::runtime_error("Unsupported data type for sum");
    }
}

} // namespace nntile::graph::tensor
