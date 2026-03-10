/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/add_slice.cc
 * TensorGraph add_slice operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/add_slice.hh"

#include <stdexcept>
#include <utility>

#include "nntile/base_types.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/add_slice.hh"

namespace nntile::graph::tensor
{

namespace
{

template<typename T>
void run_add_slice(
    TensorGraph::Runtime& runtime,
    Scalar alpha, Scalar beta,
    Index axis,
    TensorGraph::TensorNode* src1,
    TensorGraph::TensorNode* src2,
    TensorGraph::TensorNode* dst)
{
    auto& src1_t = runtime.get_tensor<T>(src1);
    auto& src2_t = runtime.get_tensor<T>(src2);
    auto& dst_t = runtime.get_tensor<T>(dst);
    nntile::tensor::add_slice<T>(alpha, src1_t, beta, src2_t, dst_t, axis);
}

} // namespace

TensorGraph::TensorNode* add_slice(
    Scalar alpha,
    TensorGraph::TensorNode* src1,
    Scalar beta,
    TensorGraph::TensorNode* src2,
    const std::string& output_name,
    Index axis)
{
    if(src1 == nullptr || src2 == nullptr)
    {
        throw std::invalid_argument(
            "add_slice: input tensors must be non-null");
    }
    if(src1->graph() != src2->graph())
    {
        throw std::invalid_argument(
            "add_slice: input tensors must belong to the same graph");
    }
    if(src1->dtype() != src2->dtype())
    {
        throw std::invalid_argument(
            "add_slice: input tensors must have the same dtype");
    }
    if(src1->ndim() + 1 != src2->ndim())
    {
        throw std::invalid_argument(
            "add_slice: src1 must have ndim = src2.ndim - 1");
    }
    if(axis < 0 || axis >= src2->ndim())
    {
        throw std::invalid_argument(
            "add_slice: axis out of range");
    }

    // Merge slice broadcast: src1 with src2
    int d = 0;
    for(Index i = 0; i < src2->ndim(); ++i)
    {
        if(i == axis) continue;
        merge_axis(src1->mutable_axes()[static_cast<size_t>(d)],
                   src2->mutable_axes()[static_cast<size_t>(i)]);
        ++d;
    }

    // Output shape matches src2
    std::vector<Index> output_shape = src2->shape();
    TensorGraph::TensorNode* output = src2->graph()->data(
        std::move(output_shape),
        output_name,
        src2->dtype());
    output->set_axes(src2->axes());

    auto op = std::make_shared<TensorAddSliceOp>(
        src1, src2, output, alpha, beta, axis);
    src1->graph()->add_op(op);

    return output;
}

void add_slice(
    Scalar alpha,
    TensorGraph::TensorNode* src1,
    Scalar beta,
    TensorGraph::TensorNode* src2,
    TensorGraph::TensorNode* dst,
    Index axis)
{
    if(src1 == nullptr || src2 == nullptr || dst == nullptr)
    {
        throw std::invalid_argument(
            "add_slice: input tensors must be non-null");
    }
    if(src1->graph() != src2->graph() || src1->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "add_slice: input tensors must belong to the same graph");
    }
    if(src1->dtype() != src2->dtype() || src1->dtype() != dst->dtype())
    {
        throw std::invalid_argument(
            "add_slice: input tensors must have the same dtype");
    }
    if(src1 == src2 || src1 == dst || src2 == dst)
    {
        throw std::invalid_argument(
            "add_slice: src1, src2, and dst must be distinct tensors");
    }
    if(src1->ndim() + 1 != src2->ndim())
    {
        throw std::invalid_argument(
            "add_slice: src1 must have ndim = src2.ndim - 1");
    }

    // Merge slice broadcast: src1 with src2
    {
        int d = 0;
        for(Index i = 0; i < src2->ndim(); ++i)
        {
            if(i == axis) continue;
            merge_axis(src1->mutable_axes()[static_cast<size_t>(d)],
                       src2->mutable_axes()[static_cast<size_t>(i)]);
            ++d;
        }
    }
    // Merge src2 with dst (same shape)
    if(src2->ndim() != dst->ndim())
    {
        throw std::invalid_argument(
            "add_slice: dst ndim must match src2 ndim");
    }
    for(Index i = 0; i < src2->ndim(); ++i)
    {
        merge_axis(src2->mutable_axes()[i], dst->mutable_axes()[i]);
    }

    auto op = std::make_shared<TensorAddSliceOp>(
        src1, src2, dst, alpha, beta, axis);
    src1->graph()->add_op(op);
}

void TensorAddSliceOp::execute(
    TensorGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(src1);

    switch(dtype)
    {
        case DataType::FP32:
            run_add_slice<nntile::fp32_t>(
                runtime, alpha, beta, axis, src1, src2, dst);
            break;
        case DataType::FP32_FAST_TF32:
            run_add_slice<nntile::fp32_fast_tf32_t>(
                runtime, alpha, beta, axis, src1, src2, dst);
            break;
        case DataType::FP32_FAST_FP16:
            run_add_slice<nntile::fp32_fast_fp16_t>(
                runtime, alpha, beta, axis, src1, src2, dst);
            break;
        case DataType::FP32_FAST_BF16:
            run_add_slice<nntile::fp32_fast_bf16_t>(
                runtime, alpha, beta, axis, src1, src2, dst);
            break;
        case DataType::FP64:
            run_add_slice<nntile::fp64_t>(
                runtime, alpha, beta, axis, src1, src2, dst);
            break;
        case DataType::FP16:
            run_add_slice<nntile::fp16_t>(
                runtime, alpha, beta, axis, src1, src2, dst);
            break;
        case DataType::BF16:
            run_add_slice<nntile::bf16_t>(
                runtime, alpha, beta, axis, src1, src2, dst);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for add_slice operation");
        default:
            throw std::runtime_error("Unsupported data type for add_slice");
    }
}

} // namespace nntile::graph::tensor
