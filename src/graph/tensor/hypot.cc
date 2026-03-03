/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/hypot.cc
 * TensorGraph hypot operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/hypot.hh"

#include <stdexcept>
#include <utility>

#include "nntile/base_types.hh"
#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/hypot.hh"

namespace nntile::graph
{

namespace
{

template<typename T>
void run_hypot(
    TensorGraph::Runtime& runtime,
    Scalar alpha, Scalar beta,
    TensorGraph::TensorNode* src1,
    TensorGraph::TensorNode* src2,
    TensorGraph::TensorNode* dst)
{
    auto& src1_t = runtime.get_tensor<T>(src1);
    auto& src2_t = runtime.get_tensor<T>(src2);
    auto& dst_t = runtime.get_tensor<T>(dst);
    nntile::tensor::hypot<T>(alpha, src1_t, beta, src2_t, dst_t);
}

} // namespace

TensorGraph::TensorNode* hypot(
    Scalar alpha,
    TensorGraph::TensorNode* src1,
    Scalar beta,
    TensorGraph::TensorNode* src2,
    const std::string& output_name)
{
    if(src1 == nullptr || src2 == nullptr)
    {
        throw std::invalid_argument(
            "hypot: input tensors must be non-null");
    }
    if(src1->graph() != src2->graph())
    {
        throw std::invalid_argument(
            "hypot: input tensors must belong to the same graph");
    }
    if(src1->dtype() != src2->dtype())
    {
        throw std::invalid_argument(
            "hypot: input tensors must have the same dtype");
    }
    if(src1->shape() != src2->shape())
    {
        throw std::invalid_argument(
            "hypot: input tensors must have the same shape");
    }

    std::vector<Index> output_shape = src1->shape();
    TensorGraph::TensorNode* dst = src1->graph()->data(
        std::move(output_shape),
        output_name,
        src1->dtype());

    hypot(alpha, src1, beta, src2, dst);

    return dst;
}

void hypot(
    Scalar alpha,
    TensorGraph::TensorNode* src1,
    Scalar beta,
    TensorGraph::TensorNode* src2,
    TensorGraph::TensorNode* dst)
{
    if(src1 == nullptr || src2 == nullptr || dst == nullptr)
    {
        throw std::invalid_argument(
            "hypot: input tensors must be non-null");
    }
    if(src1->graph() != src2->graph() || src1->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "hypot: input tensors must belong to the same graph");
    }
    if(src1->dtype() != src2->dtype() || src1->dtype() != dst->dtype())
    {
        throw std::invalid_argument(
            "hypot: input tensors must have the same dtype");
    }
    if(src1->shape() != dst->shape())
    {
        throw std::invalid_argument(
            "hypot: output shape must match input shape");
    }

    auto op = std::make_shared<TensorHypotOp>(
        alpha, beta, src1, src2, dst);
    src1->graph()->add_op(op);
}

void TensorHypotOp::execute(
    TensorGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(src1);

    switch(dtype)
    {
        case DataType::FP32:
            run_hypot<nntile::fp32_t>(runtime, alpha, beta, src1, src2, dst);
            break;
        case DataType::FP32_FAST_TF32:
            run_hypot<nntile::fp32_fast_tf32_t>(runtime, alpha, beta, src1, src2, dst);
            break;
        case DataType::FP32_FAST_FP16:
            run_hypot<nntile::fp32_fast_fp16_t>(runtime, alpha, beta, src1, src2, dst);
            break;
        case DataType::FP32_FAST_BF16:
            run_hypot<nntile::fp32_fast_bf16_t>(runtime, alpha, beta, src1, src2, dst);
            break;
        case DataType::FP64:
            run_hypot<nntile::fp64_t>(runtime, alpha, beta, src1, src2, dst);
            break;
        case DataType::FP16:
            run_hypot<nntile::fp16_t>(runtime, alpha, beta, src1, src2, dst);
            break;
        case DataType::BF16:
            run_hypot<nntile::bf16_t>(runtime, alpha, beta, src1, src2, dst);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for hypot operation");
        default:
            throw std::runtime_error("Unsupported data type for hypot");
    }
}

} // namespace nntile::graph
