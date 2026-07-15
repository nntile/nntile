#include <nntile/common.hh>
/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/tensor_graph/flash_sdpa_fwd_cudnn.cc
 * TensorGraph flash_sdpa_fwd_cudnn operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/ops/flash_sdpa_fwd_cudnn.hh"

#include "nntile/base_types.hh"
#include "nntile/tensor.hh"
#include "nntile/tensor/tensor_graph_tiling.hh"
#include "nntile/tensor/tile_lowering_helpers.hh"
#include "nntile/tile/lowering_context.hh"
#include "nntile/tile/ops/flash_sdpa_fwd_cudnn.hh"
#include "nntile/tile/ops/clear.hh"
#include "nntile/tile/ops/fill.hh"
#include "nntile/tensor/ops/flash_sdpa_fwd_cudnn.hh"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace nntile::tensor
{

TensorGraph::TensorNode *flash_sdpa_fwd_cudnn(TensorGraph::TensorNode *K,
    TensorGraph::TensorNode *Q,
    TensorGraph::TensorNode *mask,
    TensorGraph::TensorNode *V,
    const std::string &logsumexp_name)
{
    if (K == nullptr || Q == nullptr || mask == nullptr || V == nullptr)
        throw std::invalid_argument(
            "flash_sdpa_fwd_cudnn: tensors must be non-null");
    if (K->graph() != Q->graph() || Q->graph() != mask->graph() ||
        mask->graph() != V->graph())
        throw std::invalid_argument(
            "flash_sdpa_fwd_cudnn: tensors must belong to same graph");
    if (K->dtype() != Q->dtype() || Q->dtype() != mask->dtype() ||
        mask->dtype() != V->dtype())
        throw std::invalid_argument(
            "flash_sdpa_fwd_cudnn: K, Q, mask, V must have same dtype");
    if (K->ndim() != 5 || Q->ndim() != 5)
        throw std::invalid_argument(
            "flash_sdpa_fwd_cudnn: K and Q must be 5D");
    // logsumexp is FP32 (4D: seq, batch, kv_group_size, n_head_kv)
    // A is 5D like Q (head_size, seq, batch, kv_group_size, n_head_kv)
    const auto &Q_shape = Q->shape();
    std::vector<Index> logsumexp_shape(Q_shape.begin() + 1, Q_shape.end());
    std::vector<Index> A_shape = Q_shape;
    TensorGraph::TensorNode *logsumexp_node =
        K->graph()->emplace_data(std::move(logsumexp_shape), DataType::FP32);
    logsumexp_node->set_name(logsumexp_name);
    TensorGraph::TensorNode *A_node =
        K->graph()->emplace_data(std::move(A_shape), K->dtype());

    // A has same shape as Q
    A_node->set_axes(Q->axes());
    // logsumexp drops head_size (axis 0)
    for (Index i = 0; i < logsumexp_node->ndim(); ++i)
    {
        merge_axis(
            logsumexp_node->mutable_axes()[i], Q->mutable_axes()[i + 1]);
    }

    flash_sdpa_fwd_cudnn(K, Q, mask, logsumexp_node, V, A_node);
    return A_node;
}

void flash_sdpa_fwd_cudnn(TensorGraph::TensorNode *K,
    TensorGraph::TensorNode *Q,
    TensorGraph::TensorNode *mask,
    TensorGraph::TensorNode *logsumexp,
    TensorGraph::TensorNode *V,
    TensorGraph::TensorNode *A)
{
    if (K == nullptr || Q == nullptr || mask == nullptr ||
        logsumexp == nullptr || V == nullptr || A == nullptr)
        throw std::invalid_argument(
            "flash_sdpa_fwd_cudnn: tensors must be non-null");
    if (K->graph() != Q->graph() || Q->graph() != mask->graph() ||
        mask->graph() != logsumexp->graph() ||
        logsumexp->graph() != V->graph() || V->graph() != A->graph())
        throw std::invalid_argument(
            "flash_sdpa_fwd_cudnn: tensors must belong to same graph");
    if (K->dtype() != Q->dtype() || Q->dtype() != mask->dtype() ||
        mask->dtype() != V->dtype() || V->dtype() != A->dtype())
        throw std::invalid_argument(
            "flash_sdpa_fwd_cudnn: K, Q, mask, V, A must have same dtype");
    if (logsumexp->dtype() != DataType::FP32)
        throw std::invalid_argument(
            "flash_sdpa_fwd_cudnn: logsumexp must have FP32 dtype");
    validate_same_shape_and_merge(Q, A, "flash_sdpa_fwd_cudnn");
    validate_logsumexp_drop_first_shape_and_merge(
        Q, logsumexp, "flash_sdpa_fwd_cudnn");
    validate_flash_sdpa_qkv_shape_and_merge(Q, K, V, "flash_sdpa_fwd_cudnn");
    if (mask->ndim() != 2)
        throw std::invalid_argument("flash_sdpa_fwd_cudnn: mask must be 2D");
    const Index q_seq_ax = 1;
    const Index k_seq_ax = 1;
    if (mask->shape()[0] != Q->shape()[q_seq_ax] ||
        mask->shape()[1] != K->shape()[k_seq_ax])
        throw std::invalid_argument(
            "flash_sdpa_fwd_cudnn: mask shape must be {Q_seq, K_seq}");
    merge_axis(mask->mutable_axes()[0], Q->mutable_axes()[q_seq_ax]);
    merge_axis(mask->mutable_axes()[1], K->mutable_axes()[k_seq_ax]);

    auto op = std::make_shared<TensorFlashSdpaFwdCudnnOp>(
        K, Q, mask, logsumexp, V, A);
    A->graph()->add_op(op);
}

void TensorFlashSdpaFwdCudnnOp::lower_to_tile(const LoweringContext &ctx) const
{
    constexpr const char *op = "FLASH_SDPA_FWD_CUDNN";
    const TensorAxisLayout *lay_k = ctx.tiling.find(K);
    const TensorAxisLayout *lay_q = ctx.tiling.find(Q);
    const TensorAxisLayout *lay_v = ctx.tiling.find(V);
    const TensorAxisLayout *lay_a = ctx.tiling.find(A);
    const TensorAxisLayout *lay_mask = ctx.tiling.find(mask);
    const TensorAxisLayout *lay_lse = ctx.tiling.find(logsumexp);
    if (lay_k == nullptr || lay_q == nullptr || lay_v == nullptr ||
        lay_a == nullptr || lay_mask == nullptr || lay_lse == nullptr)
    {
        throw std::runtime_error(
            std::string("lower_to_tile ") + op +
            ": missing tiling for K/Q/V/A/mask/logsumexp");
    }
    if (lay_q->grid_shape() != lay_k->grid_shape() ||
        lay_q->grid_shape() != lay_v->grid_shape() ||
        lay_q->grid_shape() != lay_a->grid_shape())
    {
        throw std::runtime_error(
            std::string("lower_to_tile ") + op +
            ": K/Q/V/A must share the same per-axis tile grid");
    }
    if (lay_k->grid_shape()[0] != 1)
    {
        throw std::runtime_error(
            std::string("lower_to_tile ") + op +
            ": head dimension must not be tiled (first grid axis != 1)");
    }
    const Index seq_ax = 1;
    if (lay_mask->grid_shape()[0] != lay_q->grid_shape()[static_cast<size_t>(seq_ax)] ||
        lay_mask->grid_shape()[1] != lay_k->grid_shape()[static_cast<size_t>(seq_ax)])
    {
        throw std::runtime_error(
            std::string("lower_to_tile ") + op +
            ": mask tile grid must align with Q and K sequence axes");
    }
    if (lay_lse->grid_shape().size() != static_cast<size_t>(K->ndim() - 1))
    {
        throw std::runtime_error(
            std::string("lower_to_tile ") + op + ": logsumexp rank mismatch");
    }
    for (Index i = 1; i < K->ndim(); ++i)
    {
        if (lay_lse->grid_shape()[static_cast<size_t>(i - 1)] !=
            lay_q->grid_shape()[static_cast<size_t>(i)])
        {
            throw std::runtime_error(
                std::string("lower_to_tile ") + op +
                ": logsumexp tile grid must match Q leading axes");
        }
    }

    const Index num_k_seq_tiles =
        lay_k->grid_shape()[static_cast<size_t>(seq_ax)];
    const auto &tiles_k = tile_lower::tiles_of(ctx.tile_map, K);
    const auto &tiles_q = tile_lower::tiles_of(ctx.tile_map, Q);
    const auto &tiles_v = tile_lower::tiles_of(ctx.tile_map, V);
    const auto &tiles_a = tile_lower::tiles_of(ctx.tile_map, A);
    const auto &tiles_mask = tile_lower::tiles_of(ctx.tile_map, mask);
    const auto &tiles_lse = tile_lower::tiles_of(ctx.tile_map, logsumexp);

    for (Index lin_lse = 0; lin_lse < lay_lse->grid_volume(); ++lin_lse)
    {
        tile::fill(
            -std::numeric_limits<float>::infinity(),
            tiles_lse[static_cast<size_t>(lin_lse)]);
    }
    for (Index lin_a = 0; lin_a < lay_a->grid_volume(); ++lin_a)
    {
        tile::clear(tiles_a[static_cast<size_t>(lin_a)]);
    }

    std::vector<Index> a_coord(5);
    std::vector<Index> kv_coord(5);
    std::vector<Index> mask_coord(2);
    std::vector<Index> lse_coord(4);

    for (Index lin_a = 0; lin_a < lay_a->grid_volume(); ++lin_a)
    {
        lay_a->grid_coord_from_linear(lin_a, a_coord);
        for (Index i = 1; i < K->ndim(); ++i)
        {
            lse_coord[static_cast<size_t>(i - 1)] =
                a_coord[static_cast<size_t>(i)];
        }
        const Index lin_lse = lay_lse->grid_linear(lse_coord);

        for (Index k_seq_idx = 0; k_seq_idx < num_k_seq_tiles; ++k_seq_idx)
        {
            kv_coord = a_coord;
            kv_coord[static_cast<size_t>(seq_ax)] = k_seq_idx;
            const Index lin_kv = lay_k->grid_linear(kv_coord);
            mask_coord[0] = a_coord[static_cast<size_t>(seq_ax)];
            mask_coord[1] = k_seq_idx;
            const Index lin_mask = lay_mask->grid_linear(mask_coord);

            tile::flash_sdpa_fwd_cudnn(
                tiles_k[static_cast<size_t>(lin_kv)],
                tiles_q[static_cast<size_t>(lin_a)],
                tiles_mask[static_cast<size_t>(lin_mask)],
                tiles_lse[static_cast<size_t>(lin_lse)],
                tiles_v[static_cast<size_t>(lin_kv)],
                tiles_a[static_cast<size_t>(lin_a)]);
        }
    }
}

} // namespace nntile::tensor
