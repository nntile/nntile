/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/scale_inplace.cc
 * TensorGraph scale_inplace operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/scale_inplace.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/scale_inplace.hh"

namespace nntile::graph
{

namespace
{

template<typename T>
void run_scale_inplace(TensorGraph::Runtime& runtime, Scalar alpha,
                      TensorGraph::TensorNode* dst)
{
    auto& dst_t = runtime.get_tensor<T>(dst);
    nntile::tensor::scale_inplace<T>(alpha, dst_t);
}

} // namespace

void scale_inplace(Scalar alpha, TensorGraph::TensorNode* dst)
{
    if(dst == nullptr)
        throw std::invalid_argument("scale_inplace: tensor must be non-null");
    auto op = std::make_shared<TensorScaleInplaceOp>(alpha, dst);
    dst->graph()->add_op(op);
}

void TensorScaleInplaceOp::execute(
    TensorGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(dst);

    switch(dtype)
    {
        case DataType::FP32:
            run_scale_inplace<nntile::fp32_t>(runtime, alpha, dst);
            break;
        case DataType::FP32_FAST_TF32:
            run_scale_inplace<nntile::fp32_fast_tf32_t>(runtime, alpha, dst);
            break;
        case DataType::FP32_FAST_FP16:
            run_scale_inplace<nntile::fp32_fast_fp16_t>(runtime, alpha, dst);
            break;
        case DataType::FP32_FAST_BF16:
            run_scale_inplace<nntile::fp32_fast_bf16_t>(runtime, alpha, dst);
            break;
        case DataType::FP64:
            run_scale_inplace<nntile::fp64_t>(runtime, alpha, dst);
            break;
        case DataType::FP16:
            run_scale_inplace<nntile::fp16_t>(runtime, alpha, dst);
            break;
        case DataType::BF16:
            run_scale_inplace<nntile::bf16_t>(runtime, alpha, dst);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " not supported for scale_inplace");
        default:
            throw std::runtime_error("Unsupported data type for scale_inplace");
    }
}

} // namespace nntile::graph
