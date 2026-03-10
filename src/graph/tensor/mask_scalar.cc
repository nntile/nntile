/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/mask_scalar.cc
 * TensorGraph mask_scalar operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/mask_scalar.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/mask_scalar.hh"

namespace nntile::graph::tensor
{

namespace
{

template<typename T>
void run_mask_scalar(TensorGraph::Runtime& runtime,
                     TensorGraph::TensorNode* mask,
                     Scalar val,
                     TensorGraph::TensorNode* A,
                     Index batch_ndim)
{
    auto& mask_t = runtime.get_tensor<nntile::bool_t>(mask);
    auto& A_t = runtime.get_tensor<T>(A);
    nntile::tensor::mask_scalar<T>(mask_t, val, A_t, batch_ndim);
}

} // namespace

void mask_scalar(TensorGraph::TensorNode* mask,
                 Scalar val,
                 TensorGraph::TensorNode* A,
                 Index batch_ndim)
{
    if(mask == nullptr || A == nullptr)
        throw std::invalid_argument("mask_scalar: tensors must be non-null");
    if(mask->graph() != A->graph())
        throw std::invalid_argument("mask_scalar: tensors must belong to same graph");
    if(mask->dtype() != DataType::BOOL)
        throw std::invalid_argument("mask_scalar: mask must have BOOL dtype");
    Index A_data_ndim = A->ndim() - batch_ndim;
    if(mask->ndim() != A_data_ndim)
    {
        throw std::invalid_argument(
            "mask_scalar: mask.ndim must equal A.ndim - batch_ndim (" +
            std::to_string(mask->ndim()) + " vs " +
            std::to_string(A_data_ndim) + ")");
    }
    for(Index i = 0; i < A_data_ndim; ++i)
    {
        if(mask->shape()[i] != A->shape()[i])
        {
            throw std::invalid_argument(
                "mask_scalar: mask.dim[" + std::to_string(i) +
                "] must match A.dim[" + std::to_string(i) + "] (" +
                std::to_string(mask->shape()[i]) + " vs " +
                std::to_string(A->shape()[i]) + ")");
        }
        merge_axis(mask->mutable_axes()[i], A->mutable_axes()[i]);
    }

    auto op = std::make_shared<TensorMaskScalarOp>(mask, val, A, batch_ndim);
    A->graph()->add_op(op);
}

void TensorMaskScalarOp::execute(TensorGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(A);
    switch(dtype)
    {
        case DataType::FP32:
            run_mask_scalar<nntile::fp32_t>(runtime, mask, val, A, batch_ndim);
            break;
        case DataType::FP32_FAST_TF32:
            run_mask_scalar<nntile::fp32_fast_tf32_t>(runtime, mask, val, A, batch_ndim);
            break;
        case DataType::FP32_FAST_FP16:
            run_mask_scalar<nntile::fp32_fast_fp16_t>(runtime, mask, val, A, batch_ndim);
            break;
        case DataType::FP32_FAST_BF16:
            run_mask_scalar<nntile::fp32_fast_bf16_t>(runtime, mask, val, A, batch_ndim);
            break;
        case DataType::FP64:
            run_mask_scalar<nntile::fp64_t>(runtime, mask, val, A, batch_ndim);
            break;
        case DataType::FP16:
            run_mask_scalar<nntile::fp16_t>(runtime, mask, val, A, batch_ndim);
            break;
        case DataType::BF16:
            run_mask_scalar<nntile::bf16_t>(runtime, mask, val, A, batch_ndim);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " not supported for mask_scalar (A tensor)");
        default:
            throw std::runtime_error("Unsupported data type for mask_scalar");
    }
}

} // namespace nntile::graph::tensor
