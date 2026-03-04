/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/transpose.cc
 * TensorGraph transpose operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/transpose.hh"

#include <stdexcept>
#include <utility>

#include "nntile/base_types.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/transpose.hh"

namespace nntile::graph::tensor
{

namespace
{

template<typename T>
void run_transpose(TensorGraph::Runtime& runtime, Scalar alpha,
                  Index ndim,
                  TensorGraph::TensorNode* src,
                  TensorGraph::TensorNode* dst)
{
    auto& src_t = runtime.get_tensor<T>(src);
    auto& dst_t = runtime.get_tensor<T>(dst);
    nntile::tensor::transpose<T>(alpha, src_t, dst_t, ndim);
}

} // namespace

TensorGraph::TensorNode* transpose(
    Scalar alpha,
    TensorGraph::TensorNode* src,
    const std::string& output_name,
    Index ndim)
{
    if(src == nullptr)
        throw std::invalid_argument("transpose: input tensor must be non-null");
    if(ndim <= 0 || ndim >= src->ndim())
        throw std::invalid_argument("transpose: ndim must be in (0, src.ndim)");
    std::vector<Index> src_shape = src->shape();
    std::vector<Index> output_shape(src->ndim());
    for(Index i = 0; i < src->ndim(); ++i)
        output_shape[i] = src_shape[(i + ndim) % src->ndim()];
    TensorGraph::TensorNode* output = src->graph()->data(
        std::move(output_shape), output_name, src->dtype());
    transpose(alpha, src, output, ndim);
    return output;
}

void transpose(
    Scalar alpha,
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst,
    Index ndim)
{
    if(src == nullptr || dst == nullptr)
        throw std::invalid_argument("transpose: tensors must be non-null");
    if(src == dst)
        throw std::invalid_argument("transpose: src and dst must be distinct tensors");
    if(src->graph() != dst->graph())
        throw std::invalid_argument("transpose: tensors must belong to same graph");
    if(src->dtype() != dst->dtype())
        throw std::invalid_argument("transpose: tensors must have same dtype");
    if(ndim <= 0 || ndim >= src->ndim())
        throw std::invalid_argument("transpose: ndim must be in (0, src.ndim)");
    if(dst->ndim() != src->ndim())
        throw std::invalid_argument("transpose: dst.ndim must equal src.ndim");
    auto src_shape = src->shape();
    for(Index i = 0; i < dst->ndim(); ++i)
    {
        if(dst->dim(i) != src_shape[(i + ndim) % dst->ndim()])
            throw std::invalid_argument("transpose: dst shape must match transpose of src");
    }
    auto op = std::make_shared<TensorTransposeOp>(src, dst, ndim, alpha);
    src->graph()->add_op(op);
}

void TensorTransposeOp::execute(TensorGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(src);
    switch(dtype)
    {
        case DataType::FP32: run_transpose<nntile::fp32_t>(runtime, alpha, ndim, src, dst); break;
        case DataType::FP32_FAST_TF32: run_transpose<nntile::fp32_fast_tf32_t>(runtime, alpha, ndim, src, dst); break;
        case DataType::FP32_FAST_FP16: run_transpose<nntile::fp32_fast_fp16_t>(runtime, alpha, ndim, src, dst); break;
        case DataType::FP32_FAST_BF16: run_transpose<nntile::fp32_fast_bf16_t>(runtime, alpha, ndim, src, dst); break;
        case DataType::FP64: run_transpose<nntile::fp64_t>(runtime, alpha, ndim, src, dst); break;
        case DataType::FP16: run_transpose<nntile::fp16_t>(runtime, alpha, ndim, src, dst); break;
        case DataType::BF16: run_transpose<nntile::bf16_t>(runtime, alpha, ndim, src, dst); break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(std::string(dtype_to_string(dtype)) +
                " not supported for transpose");
        default: throw std::runtime_error("Unsupported data type for transpose");
    }
}

} // namespace nntile::graph::tensor
