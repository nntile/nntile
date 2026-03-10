/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/flash_sdpa_bwd_cudnn.cc
 * TensorGraph flash_sdpa_bwd_cudnn operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/flash_sdpa_bwd_cudnn.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/flash_sdpa_bwd_cudnn.hh"

namespace nntile::graph::tensor
{

namespace
{

template<typename T>
void run_flash_sdpa_bwd_cudnn(TensorGraph::Runtime& runtime,
                              TensorGraph::TensorNode* K,
                              TensorGraph::TensorNode* Q,
                              TensorGraph::TensorNode* V,
                              TensorGraph::TensorNode* A,
                              TensorGraph::TensorNode* dA,
                              TensorGraph::TensorNode* mask,
                              TensorGraph::TensorNode* logsumexp,
                              TensorGraph::TensorNode* dK,
                              TensorGraph::TensorNode* dQ,
                              TensorGraph::TensorNode* dV)
{
    auto& K_t = runtime.get_tensor<T>(K);
    auto& Q_t = runtime.get_tensor<T>(Q);
    auto& V_t = runtime.get_tensor<T>(V);
    auto& A_t = runtime.get_tensor<T>(A);
    auto& dA_t = runtime.get_tensor<T>(dA);
    auto& mask_t = runtime.get_tensor<T>(mask);
    auto& logsumexp_t = runtime.get_tensor<nntile::fp32_t>(logsumexp);
    auto& dK_t = runtime.get_tensor<T>(dK);
    auto& dQ_t = runtime.get_tensor<T>(dQ);
    auto& dV_t = runtime.get_tensor<T>(dV);
    nntile::tensor::flash_sdpa_bwd_cudnn<T>(
        K_t, Q_t, V_t, A_t, dA_t, mask_t, logsumexp_t,
        dK_t, dQ_t, dV_t);
}

} // namespace

void flash_sdpa_bwd_cudnn(TensorGraph::TensorNode* K,
                          TensorGraph::TensorNode* Q,
                          TensorGraph::TensorNode* V,
                          TensorGraph::TensorNode* A,
                          TensorGraph::TensorNode* dA,
                          TensorGraph::TensorNode* mask,
                          TensorGraph::TensorNode* logsumexp,
                          TensorGraph::TensorNode* dK,
                          TensorGraph::TensorNode* dQ,
                          TensorGraph::TensorNode* dV)
{
    if(K == nullptr || Q == nullptr || V == nullptr || A == nullptr ||
       dA == nullptr || mask == nullptr || logsumexp == nullptr ||
       dK == nullptr || dQ == nullptr || dV == nullptr)
        throw std::invalid_argument(
            "flash_sdpa_bwd_cudnn: tensors must be non-null");
    if(K->graph() != Q->graph() || Q->graph() != V->graph() ||
       V->graph() != A->graph() || A->graph() != dA->graph() ||
       dA->graph() != mask->graph() || mask->graph() != logsumexp->graph() ||
       logsumexp->graph() != dK->graph() || dK->graph() != dQ->graph() ||
       dQ->graph() != dV->graph())
        throw std::invalid_argument(
            "flash_sdpa_bwd_cudnn: tensors must belong to same graph");
    if(K->dtype() != Q->dtype() || Q->dtype() != V->dtype() ||
       V->dtype() != A->dtype() || A->dtype() != dA->dtype() ||
       dA->dtype() != mask->dtype() || mask->dtype() != dK->dtype() ||
       dK->dtype() != dQ->dtype() || dQ->dtype() != dV->dtype())
        throw std::invalid_argument(
            "flash_sdpa_bwd_cudnn: K,Q,V,A,dA,mask,dK,dQ,dV must have same dtype");
    if(logsumexp->dtype() != DataType::FP32)
        throw std::invalid_argument(
            "flash_sdpa_bwd_cudnn: logsumexp must have FP32 dtype");
    validate_same_shape_and_merge(K, dK, "flash_sdpa_bwd_cudnn");
    validate_same_shape_and_merge(Q, dQ, "flash_sdpa_bwd_cudnn");
    validate_same_shape_and_merge(V, dV, "flash_sdpa_bwd_cudnn");
    validate_same_shape_and_merge(K, A, "flash_sdpa_bwd_cudnn");
    validate_same_shape_and_merge(K, dA, "flash_sdpa_bwd_cudnn");
    validate_logsumexp_shape_and_merge(K, logsumexp, "flash_sdpa_bwd_cudnn");
    if(Q->ndim() != K->ndim())
        throw std::invalid_argument(
            "flash_sdpa_bwd_cudnn: Q.ndim must match K.ndim");
    if(V->ndim() != K->ndim())
        throw std::invalid_argument(
            "flash_sdpa_bwd_cudnn: V.ndim must match K.ndim");
    if(Q->shape()[0] != K->shape()[0])
        throw std::invalid_argument(
            "flash_sdpa_bwd_cudnn: Q.dim[0] must match K.dim[0]");
    for(Index i = 2; i < K->ndim(); ++i)
    {
        if(Q->shape()[i] != K->shape()[i])
            throw std::invalid_argument(
                "flash_sdpa_bwd_cudnn: Q.dim[" + std::to_string(i) +
                "] must match K.dim[" + std::to_string(i) + "]");
    }
    if(V->shape()[0] != K->shape()[0])
        throw std::invalid_argument(
            "flash_sdpa_bwd_cudnn: V.dim[0] must match K.dim[0]");
    for(Index i = 2; i < K->ndim(); ++i)
    {
        if(V->shape()[i] != K->shape()[i])
            throw std::invalid_argument(
                "flash_sdpa_bwd_cudnn: V.dim[" + std::to_string(i) +
                "] must match K.dim[" + std::to_string(i) + "]");
    }
    // Q shares head_size and batch dims with K (dims 0, 2, 3, 4)
    merge_axis(Q->mutable_axes()[0], K->mutable_axes()[0]);
    for(Index i = 2; i < K->ndim(); ++i)
    {
        merge_axis(Q->mutable_axes()[i], K->mutable_axes()[i]);
    }
    // V shares head_size and batch dims with K (dims 0, 2, 3, 4)
    merge_axis(V->mutable_axes()[0], K->mutable_axes()[0]);
    for(Index i = 2; i < K->ndim(); ++i)
    {
        merge_axis(V->mutable_axes()[i], K->mutable_axes()[i]);
    }

    auto op = std::make_shared<TensorFlashSdpaBwdCudnnOp>(
        K, Q, V, A, dA, mask, logsumexp, dK, dQ, dV);
    dK->graph()->add_op(op);
}

void TensorFlashSdpaBwdCudnnOp::execute(
    TensorGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(K);
    switch(dtype)
    {
        case DataType::FP16:
            run_flash_sdpa_bwd_cudnn<nntile::fp16_t>(
                runtime, K, Q, V, A, dA, mask, logsumexp, dK, dQ, dV);
            break;
        case DataType::BF16:
            run_flash_sdpa_bwd_cudnn<nntile::bf16_t>(
                runtime, K, Q, V, A, dA, mask, logsumexp, dK, dQ, dV);
            break;
        case DataType::FP32:
        case DataType::FP32_FAST_TF32:
        case DataType::FP32_FAST_FP16:
        case DataType::FP32_FAST_BF16:
        case DataType::FP64:
            throw std::runtime_error(
                "flash_sdpa_bwd_cudnn requires FP16 or BF16 (CUDA only)");
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " not supported for flash_sdpa_bwd_cudnn");
        default:
            throw std::runtime_error(
                "Unsupported data type for flash_sdpa_bwd_cudnn");
    }
}

} // namespace nntile::graph::tensor
