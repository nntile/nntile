/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/gelutanh_inplace.cc
 * TensorGraph gelutanh_inplace operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/gelutanh_inplace.hh"

#include <stdexcept>

#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/gelutanh_inplace.hh"

namespace nntile::graph::tensor
{

namespace
{

template<typename T>
void run_gelutanh_inplace(
    TensorGraph::Runtime& runtime,
    TensorGraph::TensorNode* dst)
{
    auto& dst_t = runtime.get_tensor<T>(dst);
    nntile::tensor::gelutanh_inplace<T>(dst_t);
}

} // namespace

void gelutanh_inplace(TensorGraph::TensorNode* dst)
{
    if(dst == nullptr)
    {
        throw std::invalid_argument(
            "gelutanh_inplace: dst tensor must be non-null");
    }

    auto op = std::make_shared<TensorGelutanhInplaceOp>(dst);
    dst->graph()->add_op(op);
}

void TensorGelutanhInplaceOp::execute(
    TensorGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(dst);

    switch(dtype)
    {
        case DataType::FP32:
            run_gelutanh_inplace<nntile::fp32_t>(runtime, dst);
            break;
        case DataType::FP32_FAST_TF32:
            run_gelutanh_inplace<nntile::fp32_fast_tf32_t>(runtime, dst);
            break;
        case DataType::FP32_FAST_FP16:
            run_gelutanh_inplace<nntile::fp32_fast_fp16_t>(runtime, dst);
            break;
        case DataType::FP32_FAST_BF16:
            run_gelutanh_inplace<nntile::fp32_fast_bf16_t>(runtime, dst);
            break;
        case DataType::FP64:
            run_gelutanh_inplace<nntile::fp64_t>(runtime, dst);
            break;
        case DataType::FP16:
            throw std::runtime_error(
                "FP16 data type not supported for gelutanh_inplace operation");
        case DataType::BF16:
            run_gelutanh_inplace<nntile::bf16_t>(runtime, dst);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for gelutanh_inplace operation");
        default:
            throw std::runtime_error(
                "Unsupported data type for gelutanh_inplace");
    }
}

} // namespace nntile::graph::tensor
