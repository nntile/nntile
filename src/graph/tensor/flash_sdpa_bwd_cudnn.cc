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
#include <vector>

#include "nntile/base_types.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/graph/tensor/tensor_graph_tiling.hh"
#include "nntile/graph/tensor/tile_lowering_helpers.hh"
#include "nntile/graph/tile/flash_sdpa_bwd_cudnn.hh"
#include "nntile/graph/tile/lowering_context.hh"
#include "nntile/tensor/flash_sdpa_bwd_cudnn.hh"

namespace nntile::graph::tensor
{



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
    validate_same_shape_and_merge(Q, A, "flash_sdpa_bwd_cudnn");
    validate_same_shape_and_merge(Q, dA, "flash_sdpa_bwd_cudnn");
    validate_logsumexp_shape_and_merge(Q, logsumexp, "flash_sdpa_bwd_cudnn");
    validate_flash_sdpa_qkv_shape_and_merge(Q, K, V, "flash_sdpa_bwd_cudnn");
    if(mask->ndim() != 2)
        throw std::invalid_argument(
            "flash_sdpa_bwd_cudnn: mask must be 2D");
    if(mask->shape()[0] != K->shape()[1] || mask->shape()[1] != Q->shape()[1])
        throw std::invalid_argument(
            "flash_sdpa_bwd_cudnn: mask shape must be {K_seq, Q_seq}");
    merge_axis(mask->mutable_axes()[0], K->mutable_axes()[1]);
    merge_axis(mask->mutable_axes()[1], Q->mutable_axes()[1]);

    auto op = std::make_shared<TensorFlashSdpaBwdCudnnOp>(
        K, Q, V, A, dA, mask, logsumexp, dK, dQ, dV);
    dK->graph()->add_op(op);
}

void TensorFlashSdpaBwdCudnnOp::lower_to_tile(const LoweringContext& ctx) const
{
    constexpr const char* op = "FLASH_SDPA_BWD_CUDNN";
    const TensorAxisLayout* lay_k = ctx.tiling.find(K);
    const TensorAxisLayout* lay_q = ctx.tiling.find(Q);
    const TensorAxisLayout* lay_v = ctx.tiling.find(V);
    const TensorAxisLayout* lay_a = ctx.tiling.find(A);
    const TensorAxisLayout* lay_da = ctx.tiling.find(dA);
    const TensorAxisLayout* lay_mask = ctx.tiling.find(mask);
    const TensorAxisLayout* lay_lse = ctx.tiling.find(logsumexp);
    const TensorAxisLayout* lay_dk = ctx.tiling.find(dK);
    const TensorAxisLayout* lay_dq = ctx.tiling.find(dQ);
    const TensorAxisLayout* lay_dv = ctx.tiling.find(dV);
    if(lay_k == nullptr || lay_q == nullptr || lay_v == nullptr
        || lay_a == nullptr || lay_da == nullptr || lay_mask == nullptr
        || lay_lse == nullptr || lay_dk == nullptr || lay_dq == nullptr
        || lay_dv == nullptr)
    {
        throw std::runtime_error(std::string("lower_to_tile ") + op +
            ": missing tiling for an input/output tensor");
    }
    if(lay_dq->grid_shape() != lay_q->grid_shape()
        || lay_dq->grid_shape() != lay_k->grid_shape()
        || lay_dq->grid_shape() != lay_v->grid_shape()
        || lay_dq->grid_shape() != lay_a->grid_shape()
        || lay_dq->grid_shape() != lay_da->grid_shape()
        || lay_dq->grid_shape() != lay_dk->grid_shape()
        || lay_dq->grid_shape() != lay_dv->grid_shape())
    {
        throw std::runtime_error(std::string("lower_to_tile ") + op +
            ": K/Q/V/A/dA/dK/dQ/dV must share the same per-axis tile grid");
    }
    if(lay_k->grid_shape()[0] != 1)
    {
        throw std::runtime_error(std::string("lower_to_tile ") + op +
            ": head dimension must not be tiled (grid_shape[0] != 1)");
    }
    if(lay_mask->grid_shape()[0] != lay_k->grid_shape()[1]
        || lay_mask->grid_shape()[1] != lay_q->grid_shape()[1])
    {
        throw std::runtime_error(std::string("lower_to_tile ") + op +
            ": mask tile grid must align with K dim1 and Q dim1");
    }
    for(int i = 0; i < 4; ++i)
    {
        if(lay_lse->grid_shape()[static_cast<size_t>(i)]
            != lay_q->grid_shape()[static_cast<size_t>(i + 1)])
        {
            throw std::runtime_error(std::string("lower_to_tile ") + op +
                ": logsumexp tile grid must match Q on tail axes");
        }
    }

    const Index num_k_seq_tiles = lay_k->grid_shape()[1];
    const auto& tiles_k = tile_lower::tiles_of(ctx.tile_map, K);
    const auto& tiles_q = tile_lower::tiles_of(ctx.tile_map, Q);
    const auto& tiles_v = tile_lower::tiles_of(ctx.tile_map, V);
    const auto& tiles_a = tile_lower::tiles_of(ctx.tile_map, A);
    const auto& tiles_da = tile_lower::tiles_of(ctx.tile_map, dA);
    const auto& tiles_mask = tile_lower::tiles_of(ctx.tile_map, mask);
    const auto& tiles_lse = tile_lower::tiles_of(ctx.tile_map, logsumexp);
    const auto& tiles_dk = tile_lower::tiles_of(ctx.tile_map, dK);
    const auto& tiles_dq = tile_lower::tiles_of(ctx.tile_map, dQ);
    const auto& tiles_dv = tile_lower::tiles_of(ctx.tile_map, dV);

    std::vector<Index> dq_coord(5);
    std::vector<Index> kv_coord(5);
    std::vector<Index> mask_coord(2);
    std::vector<Index> lse_coord(4);

    for(Index lin_dq = 0; lin_dq < lay_dq->grid_volume(); ++lin_dq)
    {
        lay_dq->grid_coord_from_linear(lin_dq, dq_coord);
        for(Index i = 0; i < 4; ++i)
        {
            lse_coord[static_cast<size_t>(i)] =
                dq_coord[static_cast<size_t>(i + 1)];
        }
        const Index lin_lse = lay_lse->grid_linear(lse_coord);

        for(Index k_seq_idx = 0; k_seq_idx < num_k_seq_tiles; ++k_seq_idx)
        {
            kv_coord = dq_coord;
            kv_coord[1] = k_seq_idx;
            const Index lin_kv = lay_k->grid_linear(kv_coord);
            mask_coord[0] = k_seq_idx;
            mask_coord[1] = dq_coord[1];
            const Index lin_mask = lay_mask->grid_linear(mask_coord);

            tile_graph::flash_sdpa_bwd_cudnn(
                tiles_k[static_cast<size_t>(lin_kv)],
                tiles_q[static_cast<size_t>(lin_dq)],
                tiles_v[static_cast<size_t>(lin_kv)],
                tiles_a[static_cast<size_t>(lin_dq)],
                tiles_da[static_cast<size_t>(lin_dq)],
                tiles_mask[static_cast<size_t>(lin_mask)],
                tiles_lse[static_cast<size_t>(lin_lse)],
                tiles_dk[static_cast<size_t>(lin_kv)],
                tiles_dq[static_cast<size_t>(lin_dq)],
                tiles_dv[static_cast<size_t>(lin_kv)]);
        }
    }
}

} // namespace nntile::graph::tensor
