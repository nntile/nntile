/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/flash_sdpa_fwd_cudnn.cc
 * TensorGraph flash_sdpa_fwd_cudnn operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/flash_sdpa_fwd_cudnn.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/flash_sdpa_fwd_cudnn.hh"

namespace nntile::graph::tensor
{

namespace
{

template<typename T>
void run_flash_sdpa_fwd_cudnn(TensorGraph::Runtime& runtime,
                              TensorGraph::TensorNode* K,
                              TensorGraph::TensorNode* Q,
                              TensorGraph::TensorNode* mask,
                              TensorGraph::TensorNode* logsumexp,
                              TensorGraph::TensorNode* V,
                              TensorGraph::TensorNode* A)
{
    auto& K_t = runtime.get_tensor<T>(K);
    auto& Q_t = runtime.get_tensor<T>(Q);
    auto& mask_t = runtime.get_tensor<T>(mask);
    auto& logsumexp_t = runtime.get_tensor<nntile::fp32_t>(logsumexp);
    auto& V_t = runtime.get_tensor<T>(V);
    auto& A_t = runtime.get_tensor<T>(A);
    nntile::tensor::flash_sdpa_fwd_cudnn<T>(
        K_t, Q_t, mask_t, logsumexp_t, V_t, A_t);
}

} // namespace

TensorGraph::TensorNode* flash_sdpa_fwd_cudnn(
    TensorGraph::TensorNode* K,
    TensorGraph::TensorNode* Q,
    TensorGraph::TensorNode* mask,
    TensorGraph::TensorNode* V,
    const std::string& logsumexp_name,
    const std::string& output_name)
{
    if(K == nullptr || Q == nullptr || mask == nullptr || V == nullptr)
        throw std::invalid_argument(
            "flash_sdpa_fwd_cudnn: tensors must be non-null");
    if(K->graph() != Q->graph() || Q->graph() != mask->graph() ||
       mask->graph() != V->graph())
        throw std::invalid_argument(
            "flash_sdpa_fwd_cudnn: tensors must belong to same graph");
    if(K->dtype() != Q->dtype() || Q->dtype() != mask->dtype() ||
       mask->dtype() != V->dtype())
        throw std::invalid_argument(
            "flash_sdpa_fwd_cudnn: K, Q, mask, V must have same dtype");
    if(K->ndim() != 5)
        throw std::invalid_argument(
            "flash_sdpa_fwd_cudnn: K must be 5D");
    // logsumexp is FP32 (4D: seq, batch, kv_group_size, n_head_kv)
    // A is 5D like K (head_size, seq, batch, kv_group_size, n_head_kv)
    const auto& K_shape = K->shape();
    std::vector<Index> logsumexp_shape = {K_shape[1], K_shape[2], K_shape[3], K_shape[4]};
    std::vector<Index> A_shape = K_shape;
    TensorGraph::TensorNode* logsumexp_node = K->graph()->data(
        std::move(logsumexp_shape), logsumexp_name, DataType::FP32);
    TensorGraph::TensorNode* A_node = K->graph()->data(
        std::move(A_shape), output_name, K->dtype());

    // A has same shape as K
    A_node->set_axes(K->axes());
    // logsumexp.dim[i] == K.dim[i+1]
    for(Index i = 0; i < logsumexp_node->ndim(); ++i)
    {
        merge_axis(logsumexp_node->mutable_axes()[i],
                   K->mutable_axes()[i + 1]);
    }

    flash_sdpa_fwd_cudnn(K, Q, mask, logsumexp_node, V, A_node);
    return A_node;
}

void flash_sdpa_fwd_cudnn(TensorGraph::TensorNode* K,
                          TensorGraph::TensorNode* Q,
                          TensorGraph::TensorNode* mask,
                          TensorGraph::TensorNode* logsumexp,
                          TensorGraph::TensorNode* V,
                          TensorGraph::TensorNode* A)
{
    if(K == nullptr || Q == nullptr || mask == nullptr ||
       logsumexp == nullptr || V == nullptr || A == nullptr)
        throw std::invalid_argument(
            "flash_sdpa_fwd_cudnn: tensors must be non-null");
    if(K->graph() != Q->graph() || Q->graph() != mask->graph() ||
       mask->graph() != logsumexp->graph() || logsumexp->graph() != V->graph() ||
       V->graph() != A->graph())
        throw std::invalid_argument(
            "flash_sdpa_fwd_cudnn: tensors must belong to same graph");
    if(K->dtype() != Q->dtype() || Q->dtype() != mask->dtype() ||
       mask->dtype() != V->dtype() || V->dtype() != A->dtype())
        throw std::invalid_argument(
            "flash_sdpa_fwd_cudnn: K, Q, mask, V, A must have same dtype");
    if(logsumexp->dtype() != DataType::FP32)
        throw std::invalid_argument(
            "flash_sdpa_fwd_cudnn: logsumexp must have FP32 dtype");
    validate_same_shape_and_merge(K, A, "flash_sdpa_fwd_cudnn");
    validate_logsumexp_shape_and_merge(K, logsumexp, "flash_sdpa_fwd_cudnn");
    validate_flash_sdpa_qkv_shape_and_merge(K, Q, V, "flash_sdpa_fwd_cudnn");

    auto op = std::make_shared<TensorFlashSdpaFwdCudnnOp>(
        K, Q, mask, logsumexp, V, A);
    A->graph()->add_op(op);
}

void TensorFlashSdpaFwdCudnnOp::execute(
    TensorGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(K);
    switch(dtype)
    {
        case DataType::FP16:
            run_flash_sdpa_fwd_cudnn<nntile::fp16_t>(
                runtime, K, Q, mask, logsumexp, V, A);
            break;
        case DataType::BF16:
            run_flash_sdpa_fwd_cudnn<nntile::bf16_t>(
                runtime, K, Q, mask, logsumexp, V, A);
            break;
        case DataType::FP32:
        case DataType::FP32_FAST_TF32:
        case DataType::FP32_FAST_FP16:
        case DataType::FP32_FAST_BF16:
        case DataType::FP64:
            throw std::runtime_error(
                "flash_sdpa_fwd_cudnn requires FP16 or BF16 (CUDA only)");
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " not supported for flash_sdpa_fwd_cudnn");
        default:
            throw std::runtime_error(
                "Unsupported data type for flash_sdpa_fwd_cudnn");
    }
}

} // namespace nntile::graph::tensor
