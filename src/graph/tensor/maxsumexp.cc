/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/maxsumexp.cc
 * TensorGraph maxsumexp operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/maxsumexp.hh"

#include <stdexcept>
#include <utility>

#include "nntile/base_types.hh"
#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/clear.hh"
#include "nntile/tensor/maxsumexp.hh"

namespace nntile::graph
{

namespace
{

template<typename T>
void run_maxsumexp(
    TensorGraph::Runtime& runtime,
    Index axis, int redux,
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst)
{
    auto& src_t = runtime.get_tensor<T>(src);
    auto& dst_t = runtime.get_tensor<T>(dst);
    nntile::tensor::clear<T>(dst_t);
    nntile::tensor::maxsumexp<T>(src_t, dst_t, axis, redux);
}

} // namespace

TensorGraph::TensorNode* maxsumexp(
    TensorGraph::TensorNode* src,
    const std::string& output_name,
    Index axis,
    int redux)
{
    if(src == nullptr)
    {
        throw std::invalid_argument(
            "maxsumexp: input tensor must be non-null");
    }

    // dst shape: [2] + src.shape without axis (tensor API convention)
    // dst.shape[0]=2, dst.shape[i+1]=src.shape[i] for i<axis,
    // dst.shape[i]=src.shape[i] for i>axis
    std::vector<Index> output_shape;
    output_shape.reserve(src->ndim());
    output_shape.push_back(2);
    for(Index i = 0; i < src->ndim(); ++i)
    {
        if(i != axis)
        {
            output_shape.push_back(src->shape()[i]);
        }
    }

    TensorGraph::TensorNode* dst = src->graph()->data(
        std::move(output_shape),
        output_name,
        src->dtype());

    maxsumexp(src, dst, axis, redux);

    return dst;
}

void maxsumexp(
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst,
    Index axis,
    int redux)
{
    if(src == nullptr || dst == nullptr)
    {
        throw std::invalid_argument(
            "maxsumexp: input tensors must be non-null");
    }
    if(src->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "maxsumexp: input tensors must belong to the same graph");
    }
    if(src->dtype() != dst->dtype())
    {
        throw std::invalid_argument(
            "maxsumexp: input tensors must have the same dtype");
    }

    auto op = std::make_shared<TensorMaxsumexpOp>(src, dst, axis, redux);
    src->graph()->add_op(op);
}

void TensorMaxsumexpOp::execute(
    TensorGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(src);

    switch(dtype)
    {
        case DataType::FP32:
            run_maxsumexp<nntile::fp32_t>(runtime, axis, redux, src, dst);
            break;
        case DataType::FP32_FAST_TF32:
            run_maxsumexp<nntile::fp32_fast_tf32_t>(runtime, axis, redux, src, dst);
            break;
        case DataType::FP32_FAST_FP16:
            run_maxsumexp<nntile::fp32_fast_fp16_t>(runtime, axis, redux, src, dst);
            break;
        case DataType::FP32_FAST_BF16:
            run_maxsumexp<nntile::fp32_fast_bf16_t>(runtime, axis, redux, src, dst);
            break;
        case DataType::FP64:
            run_maxsumexp<nntile::fp64_t>(runtime, axis, redux, src, dst);
            break;
        case DataType::FP16:
            run_maxsumexp<nntile::fp16_t>(runtime, axis, redux, src, dst);
            break;
        case DataType::BF16:
            run_maxsumexp<nntile::bf16_t>(runtime, axis, redux, src, dst);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for maxsumexp operation");
        default:
            throw std::runtime_error("Unsupported data type for maxsumexp");
    }
}

} // namespace nntile::graph
