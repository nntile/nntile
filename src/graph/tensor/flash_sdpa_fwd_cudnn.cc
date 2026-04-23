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

#include <cmath>
#include <limits>
#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/clear.hh"
#include "nntile/tensor/fill.hh"
#include "nntile/tensor/flash_sdpa_fwd_cudnn.hh"

namespace nntile::graph::tensor
{



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
    if(K->ndim() != 5 || Q->ndim() != 5)
        throw std::invalid_argument(
            "flash_sdpa_fwd_cudnn: K and Q must be 5D");
    // logsumexp is FP32 (4D: seq_q, batch, kv_group_size, n_head_kv)
    // A is 5D like Q (head_size, seq_q, batch, kv_group_size, n_head_kv)
    // Output has one value per query position, so shapes derive from Q.
    const auto& Q_shape = Q->shape();
    std::vector<Index> logsumexp_shape = {Q_shape[1], Q_shape[2], Q_shape[3], Q_shape[4]};
    std::vector<Index> A_shape = Q_shape;
    TensorGraph::TensorNode* logsumexp_node = K->graph()->data(
        std::move(logsumexp_shape), logsumexp_name, DataType::FP32);
    TensorGraph::TensorNode* A_node = K->graph()->data(
        std::move(A_shape), output_name, K->dtype());

    // A has same shape as Q
    A_node->set_axes(Q->axes());
    // logsumexp.dim[i] == Q.dim[i+1]
    for(Index i = 0; i < logsumexp_node->ndim(); ++i)
    {
        merge_axis(logsumexp_node->mutable_axes()[i],
                   Q->mutable_axes()[i + 1]);
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
    validate_same_shape_and_merge(Q, A, "flash_sdpa_fwd_cudnn");
    validate_logsumexp_shape_and_merge(Q, logsumexp, "flash_sdpa_fwd_cudnn");
    validate_flash_sdpa_qkv_shape_and_merge(Q, K, V, "flash_sdpa_fwd_cudnn");
    if(mask->ndim() != 2)
        throw std::invalid_argument(
            "flash_sdpa_fwd_cudnn: mask must be 2D");
    if(mask->shape()[0] != K->shape()[1] || mask->shape()[1] != Q->shape()[1])
        throw std::invalid_argument(
            "flash_sdpa_fwd_cudnn: mask shape must be {K_seq, Q_seq}");
    merge_axis(mask->mutable_axes()[0], K->mutable_axes()[1]);
    merge_axis(mask->mutable_axes()[1], Q->mutable_axes()[1]);

    auto op = std::make_shared<TensorFlashSdpaFwdCudnnOp>(
        K, Q, mask, logsumexp, V, A);
    A->graph()->add_op(op);
}

} // namespace nntile::graph::tensor
