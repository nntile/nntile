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

namespace nntile::graph
{

namespace
{

template<typename T>
void run_flash_sdpa_bwd_cudnn(TensorGraph::ExecutionContext& ctx,
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
    auto& K_t = ctx.get_tensor<T>(K);
    auto& Q_t = ctx.get_tensor<T>(Q);
    auto& V_t = ctx.get_tensor<T>(V);
    auto& A_t = ctx.get_tensor<T>(A);
    auto& dA_t = ctx.get_tensor<T>(dA);
    auto& mask_t = ctx.get_tensor<T>(mask);
    auto& logsumexp_t = ctx.get_tensor<nntile::fp32_t>(logsumexp);
    auto& dK_t = ctx.get_tensor<T>(dK);
    auto& dQ_t = ctx.get_tensor<T>(dQ);
    auto& dV_t = ctx.get_tensor<T>(dV);
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
    auto op = std::make_shared<TensorFlashSdpaBwdCudnnOp>(
        K, Q, V, A, dA, mask, logsumexp, dK, dQ, dV);
    dK->graph()->add_op(op);
}

void TensorFlashSdpaBwdCudnnOp::execute(
    TensorGraph::ExecutionContext& ctx) const
{
    DataType dtype = ctx.get_dtype(K);
    switch(dtype)
    {
        case DataType::FP16:
            run_flash_sdpa_bwd_cudnn<nntile::fp16_t>(
                ctx, K, Q, V, A, dA, mask, logsumexp, dK, dQ, dV);
            break;
        case DataType::BF16:
            run_flash_sdpa_bwd_cudnn<nntile::bf16_t>(
                ctx, K, Q, V, A, dA, mask, logsumexp, dK, dQ, dV);
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

} // namespace nntile::graph
