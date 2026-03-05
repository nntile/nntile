/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/scale.cc
 * TensorGraph scale operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/scale.hh"

#include <stdexcept>
#include <utility>

#include "nntile/base_types.hh"
#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/scale.hh"

namespace nntile::graph::tensor
{

namespace
{

template<typename T>
void run_scale(TensorGraph::Runtime& runtime, Scalar alpha,
               TensorGraph::TensorNode* src, TensorGraph::TensorNode* dst)
{
    auto& src_t = runtime.get_tensor<T>(src);
    auto& dst_t = runtime.get_tensor<T>(dst);
    nntile::tensor::scale<T>(alpha, src_t, dst_t);
}

} // namespace

TensorGraph::TensorNode* scale(Scalar alpha, TensorGraph::TensorNode* src,
                               const std::string& output_name)
{
    if(src == nullptr)
        throw std::invalid_argument("scale: input tensor must be non-null");
    std::vector<Index> output_shape = src->shape();
    TensorGraph::TensorNode* output = src->graph()->data(
        std::move(output_shape), output_name, src->dtype());
    scale(alpha, src, output);
    return output;
}

void scale(Scalar alpha, TensorGraph::TensorNode* src,
           TensorGraph::TensorNode* dst)
{
    if(src == nullptr || dst == nullptr)
        throw std::invalid_argument("scale: tensors must be non-null");
    if(src->graph() != dst->graph())
        throw std::invalid_argument("scale: tensors must belong to same graph");
    if(src->dtype() != dst->dtype())
        throw std::invalid_argument("scale: tensors must have same dtype");
    if(src->shape() != dst->shape())
        throw std::invalid_argument("scale: tensors must have same shape");
    if(src == dst)
        throw std::invalid_argument("scale: src and dst must be distinct tensors");
    auto op = std::make_shared<TensorScaleOp>(src, dst, alpha);
    src->graph()->add_op(op);
}

void TensorScaleOp::execute(
    TensorGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(src);

    switch(dtype)
    {
        case DataType::FP32:
            run_scale<nntile::fp32_t>(runtime, alpha, src, dst);
            break;
        case DataType::FP32_FAST_TF32:
            run_scale<nntile::fp32_fast_tf32_t>(runtime, alpha, src, dst);
            break;
        case DataType::FP32_FAST_FP16:
            run_scale<nntile::fp32_fast_fp16_t>(runtime, alpha, src, dst);
            break;
        case DataType::FP32_FAST_BF16:
            run_scale<nntile::fp32_fast_bf16_t>(runtime, alpha, src, dst);
            break;
        case DataType::FP64:
            run_scale<nntile::fp64_t>(runtime, alpha, src, dst);
            break;
        case DataType::FP16:
            run_scale<nntile::fp16_t>(runtime, alpha, src, dst);
            break;
        case DataType::BF16:
            run_scale<nntile::bf16_t>(runtime, alpha, src, dst);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " not supported for scale");
        default:
            throw std::runtime_error("Unsupported data type for scale");
    }
}

} // namespace nntile::graph::tensor
