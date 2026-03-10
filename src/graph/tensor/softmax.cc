/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/softmax.cc
 * TensorGraph softmax operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/softmax.hh"

#include <stdexcept>
#include <utility>

#include "nntile/base_types.hh"
#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/softmax.hh"

namespace nntile::graph::tensor
{

namespace
{

template<typename T>
void run_softmax(
    TensorGraph::Runtime& runtime,
    Scalar alpha, Index axis,
    TensorGraph::TensorNode* maxsumexp,
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst)
{
    auto& maxsumexp_t = runtime.get_tensor<T>(maxsumexp);
    auto& src_t = runtime.get_tensor<T>(src);
    auto& dst_t = runtime.get_tensor<T>(dst);
    nntile::tensor::softmax<T>(maxsumexp_t, src_t, alpha, dst_t, axis);
}

} // namespace

TensorGraph::TensorNode* softmax(
    TensorGraph::TensorNode* maxsumexp,
    TensorGraph::TensorNode* src,
    const std::string& output_name,
    Scalar alpha,
    Index axis)
{
    if(maxsumexp == nullptr || src == nullptr)
    {
        throw std::invalid_argument(
            "softmax: input tensors must be non-null");
    }
    if(maxsumexp->graph() != src->graph())
    {
        throw std::invalid_argument(
            "softmax: input tensors must belong to the same graph");
    }
    if(maxsumexp->dtype() != src->dtype())
    {
        throw std::invalid_argument(
            "softmax: input tensors must have the same dtype");
    }
    // maxsumexp has shape with 2 at axis, src has full shape

    TensorGraph::TensorNode* dst = src->graph()->data(
        src->shape(),
        output_name,
        src->dtype());
    dst->set_axes(src->axes());

    softmax(maxsumexp, src, dst, alpha, axis);

    return dst;
}

void softmax(
    TensorGraph::TensorNode* maxsumexp,
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst,
    Scalar alpha,
    Index axis)
{
    if(maxsumexp == nullptr || src == nullptr || dst == nullptr)
    {
        throw std::invalid_argument(
            "softmax: input tensors must be non-null");
    }
    if(maxsumexp->graph() != src->graph() || maxsumexp->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "softmax: input tensors must belong to the same graph");
    }
    if(maxsumexp->dtype() != src->dtype() || maxsumexp->dtype() != dst->dtype())
    {
        throw std::invalid_argument(
            "softmax: input tensors must have the same dtype");
    }
    validate_same_shape_and_merge(src, dst, "softmax");
    validate_maxsumexp_shape_and_merge(src, maxsumexp, axis, "softmax");

    auto op = std::make_shared<TensorSoftmaxOp>(
        maxsumexp, src, dst, alpha, axis);
    src->graph()->add_op(op);
}

void TensorSoftmaxOp::execute(
    TensorGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(src);

    switch(dtype)
    {
        case DataType::FP32:
            run_softmax<nntile::fp32_t>(
                runtime, alpha, axis, maxsumexp, src, dst);
            break;
        case DataType::FP32_FAST_TF32:
            run_softmax<nntile::fp32_fast_tf32_t>(
                runtime, alpha, axis, maxsumexp, src, dst);
            break;
        case DataType::FP32_FAST_FP16:
            run_softmax<nntile::fp32_fast_fp16_t>(
                runtime, alpha, axis, maxsumexp, src, dst);
            break;
        case DataType::FP32_FAST_BF16:
            run_softmax<nntile::fp32_fast_bf16_t>(
                runtime, alpha, axis, maxsumexp, src, dst);
            break;
        case DataType::FP64:
            run_softmax<nntile::fp64_t>(
                runtime, alpha, axis, maxsumexp, src, dst);
            break;
        case DataType::FP16:
            run_softmax<nntile::fp16_t>(
                runtime, alpha, axis, maxsumexp, src, dst);
            break;
        case DataType::BF16:
            run_softmax<nntile::bf16_t>(
                runtime, alpha, axis, maxsumexp, src, dst);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for softmax operation");
        default:
            throw std::runtime_error("Unsupported data type for softmax");
    }
}

} // namespace nntile::graph::tensor
