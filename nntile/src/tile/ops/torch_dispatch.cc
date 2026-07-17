/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file src/tile/ops/torch_dispatch.cc
 * TileGraph execute for torch-native family ops.
 *
 * @version 1.1.0
 */

#include "nntile/tile/ops/torch_dispatch.hh"

#include <stdexcept>

#include <nntile/core/torch_dispatch.hh>
#include <nntile/core/torch_meta.hh>
#include <nntile/runtime.hh>

namespace nntile::tile
{

void torch_unary(
    starpu::TorchKind kind,
    TileGraph::TileNode *in,
    TileGraph::TileNode *out,
    starpu::TorchDispatchArgs extra)
{
    if (in == nullptr || out == nullptr)
    {
        throw std::invalid_argument("tile torch_unary: null tile");
    }
    auto op = std::make_shared<TileTorchUnaryOp>(kind, in, out, extra);
    in->graph()->add_op(op);
}

void torch_binary(
    starpu::TorchKind kind,
    TileGraph::TileNode *a,
    TileGraph::TileNode *b,
    TileGraph::TileNode *out,
    starpu::TorchDispatchArgs extra)
{
    if (a == nullptr || b == nullptr || out == nullptr)
    {
        throw std::invalid_argument("tile torch_binary: null tile");
    }
    auto op = std::make_shared<TileTorchBinaryOp>(
        kind,
        a,
        b,
        out,
        extra);
    a->graph()->add_op(op);
}

void torch_ternary(
    starpu::TorchKind kind,
    TileGraph::TileNode *a,
    TileGraph::TileNode *b,
    TileGraph::TileNode *c,
    TileGraph::TileNode *out,
    starpu::TorchDispatchArgs extra)
{
    if (a == nullptr || b == nullptr || c == nullptr || out == nullptr)
    {
        throw std::invalid_argument("tile torch_ternary: null tile");
    }
    auto op = std::make_shared<TileTorchTernaryOp>(
        kind,
        a,
        b,
        c,
        out,
        extra);
    a->graph()->add_op(op);
}

void torch_embedding(
    TileGraph::TileNode *weight,
    TileGraph::TileNode *indices,
    TileGraph::TileNode *out)
{
    if (weight == nullptr || indices == nullptr || out == nullptr)
    {
        throw std::invalid_argument("tile torch_embedding: null tile");
    }
    auto op = std::make_shared<TileTorchEmbeddingOp>(
        weight,
        indices,
        out);
    weight->graph()->add_op(op);
}

void torch_cat(
    Index dim,
    const std::vector<TileGraph::TileNode *> &inputs,
    TileGraph::TileNode *out)
{
    if (inputs.empty() || out == nullptr)
    {
        throw std::invalid_argument("tile torch_cat: bad args");
    }
    auto op = std::make_shared<TileTorchCatOp>(dim, inputs, out);
    inputs[0]->graph()->add_op(op);
}

void TileTorchUnaryOp::execute(Runtime &runtime) const
{
    auto &in_t = runtime.get_tile<fp32_t>(in);
    auto &out_t = runtime.get_tile<fp32_t>(out);
    const core::TorchTileMeta in_meta =
        core::meta_from_args_or_contiguous(
            extra,
            0,
            false,
            in->shape());
    const core::TorchTileMeta out_meta =
        core::meta_from_args_or_contiguous(
            extra,
            0,
            true,
            out->shape());
    core::torch_unary_out(
        runtime.starpu_worker_hint(),
        kind,
        in_t,
        in_meta,
        out_t,
        out_meta,
        extra);
}

void TileTorchBinaryOp::execute(Runtime &runtime) const
{
    auto &a_t = runtime.get_tile<fp32_t>(a);
    auto &b_t = runtime.get_tile<fp32_t>(b);
    auto &out_t = runtime.get_tile<fp32_t>(out);
    const core::TorchTileMeta a_meta =
        core::meta_from_args_or_contiguous(
            extra,
            0,
            false,
            a->shape());
    const core::TorchTileMeta b_meta =
        core::meta_from_args_or_contiguous(
            extra,
            1,
            false,
            b->shape());
    const core::TorchTileMeta out_meta =
        core::meta_from_args_or_contiguous(
            extra,
            0,
            true,
            out->shape());
    core::torch_binary_out(
        runtime.starpu_worker_hint(),
        kind,
        a_t,
        a_meta,
        b_t,
        b_meta,
        out_t,
        out_meta,
        extra);
}

void TileTorchTernaryOp::execute(Runtime &runtime) const
{
    auto &a_t = runtime.get_tile<fp32_t>(a);
    auto &b_t = runtime.get_tile<fp32_t>(b);
    auto &c_t = runtime.get_tile<fp32_t>(c);
    auto &out_t = runtime.get_tile<fp32_t>(out);
    const core::TorchTileMeta a_meta =
        core::meta_from_args_or_contiguous(
            extra,
            0,
            false,
            a->shape());
    const core::TorchTileMeta b_meta =
        core::meta_from_args_or_contiguous(
            extra,
            1,
            false,
            b->shape());
    const core::TorchTileMeta c_meta =
        core::meta_from_args_or_contiguous(
            extra,
            2,
            false,
            c->shape());
    const core::TorchTileMeta out_meta =
        core::meta_from_args_or_contiguous(
            extra,
            0,
            true,
            out->shape());
    core::torch_ternary_out(
        runtime.starpu_worker_hint(),
        kind,
        a_t,
        a_meta,
        b_t,
        b_meta,
        c_t,
        c_meta,
        out_t,
        out_meta,
        extra);
}

void TileTorchEmbeddingOp::execute(Runtime &runtime) const
{
    auto &w_t = runtime.get_tile<fp32_t>(weight);
    auto &idx_t = runtime.get_tile<int64_t>(indices);
    auto &out_t = runtime.get_tile<fp32_t>(out);
    const core::TorchTileMeta w_meta =
        core::make_contiguous_torch_meta(weight->shape());
    const core::TorchTileMeta idx_meta =
        core::make_contiguous_torch_meta(indices->shape());
    const core::TorchTileMeta out_meta =
        core::make_contiguous_torch_meta(out->shape());
    core::torch_embedding_out(
        runtime.starpu_worker_hint(),
        w_t,
        w_meta,
        idx_t,
        idx_meta,
        out_t,
        out_meta);
}

void TileTorchCatOp::execute(Runtime &runtime) const
{
    std::vector<const core::Tile<fp32_t> *> tiles;
    std::vector<core::TorchTileMeta> metas;
    tiles.reserve(inputs_tiles.size());
    metas.reserve(inputs_tiles.size());
    for (auto *n : inputs_tiles)
    {
        auto &t = runtime.get_tile<fp32_t>(n);
        tiles.push_back(&t);
        metas.push_back(core::make_contiguous_torch_meta(n->shape()));
    }
    auto &out_t = runtime.get_tile<fp32_t>(out);
    const core::TorchTileMeta out_meta =
        core::make_contiguous_torch_meta(out->shape());
    core::torch_cat_out(
        runtime.starpu_worker_hint(),
        dim,
        tiles,
        metas,
        out_t,
        out_meta);
}

void torch_layer_norm(
    TileGraph::TileNode *input,
    TileGraph::TileNode *weight,
    TileGraph::TileNode *bias,
    TileGraph::TileNode *out,
    TileGraph::TileNode *mean,
    TileGraph::TileNode *rstd,
    Index normalized_ndim,
    Scalar eps)
{
    auto op = std::make_shared<TileTorchLayerNormOp>(
        input,
        weight,
        bias,
        out,
        mean,
        rstd,
        normalized_ndim,
        eps);
    input->graph()->add_op(op);
}

void TileTorchLayerNormOp::execute(Runtime &runtime) const
{
    auto &in_t = runtime.get_tile<fp32_t>(input);
    auto &out_t = runtime.get_tile<fp32_t>(out);
    auto &mean_t = runtime.get_tile<fp32_t>(mean);
    auto &rstd_t = runtime.get_tile<fp32_t>(rstd);
    const core::TorchTileMeta in_meta =
        core::make_contiguous_torch_meta(input->shape());
    const core::TorchTileMeta out_meta =
        core::make_contiguous_torch_meta(out->shape());
    const core::TorchTileMeta mean_meta =
        core::make_contiguous_torch_meta(mean->shape());
    const core::TorchTileMeta rstd_meta =
        core::make_contiguous_torch_meta(rstd->shape());
    core::Tile<fp32_t> *w_ptr = nullptr;
    core::Tile<fp32_t> *b_ptr = nullptr;
    core::TorchTileMeta w_meta;
    core::TorchTileMeta b_meta;
    if (weight != nullptr)
    {
        w_ptr = &runtime.get_tile<fp32_t>(weight);
        w_meta = core::make_contiguous_torch_meta(weight->shape());
    }
    if (bias != nullptr)
    {
        b_ptr = &runtime.get_tile<fp32_t>(bias);
        b_meta = core::make_contiguous_torch_meta(bias->shape());
    }
    core::torch_layer_norm_out(
        runtime.starpu_worker_hint(),
        in_t,
        in_meta,
        w_ptr,
        w_ptr != nullptr ? &w_meta : nullptr,
        b_ptr,
        b_ptr != nullptr ? &b_meta : nullptr,
        out_t,
        out_meta,
        mean_t,
        mean_meta,
        rstd_t,
        rstd_meta,
        normalized_ndim,
        eps);
}

void torch_layer_norm_backward(
    TileGraph::TileNode *grad_out,
    TileGraph::TileNode *input,
    TileGraph::TileNode *mean,
    TileGraph::TileNode *rstd,
    TileGraph::TileNode *weight,
    TileGraph::TileNode *bias,
    TileGraph::TileNode *grad_input,
    TileGraph::TileNode *grad_weight,
    TileGraph::TileNode *grad_bias,
    Index normalized_ndim,
    bool need_grad_input,
    bool need_grad_weight,
    bool need_grad_bias)
{
    auto op = std::make_shared<TileTorchLayerNormBackwardOp>(
        grad_out,
        input,
        mean,
        rstd,
        weight,
        bias,
        grad_input,
        grad_weight,
        grad_bias,
        normalized_ndim,
        need_grad_input,
        need_grad_weight,
        need_grad_bias);
    grad_out->graph()->add_op(op);
}

void TileTorchLayerNormBackwardOp::execute(Runtime &runtime) const
{
    auto &go_t = runtime.get_tile<fp32_t>(grad_out);
    auto &in_t = runtime.get_tile<fp32_t>(input);
    auto &mean_t = runtime.get_tile<fp32_t>(mean);
    auto &rstd_t = runtime.get_tile<fp32_t>(rstd);
    const core::TorchTileMeta go_meta =
        core::make_contiguous_torch_meta(grad_out->shape());
    const core::TorchTileMeta in_meta =
        core::make_contiguous_torch_meta(input->shape());
    const core::TorchTileMeta mean_meta =
        core::make_contiguous_torch_meta(mean->shape());
    const core::TorchTileMeta rstd_meta =
        core::make_contiguous_torch_meta(rstd->shape());
    core::Tile<fp32_t> *w_ptr = nullptr;
    core::Tile<fp32_t> *b_ptr = nullptr;
    core::TorchTileMeta w_meta;
    core::TorchTileMeta b_meta;
    if (weight != nullptr)
    {
        w_ptr = &runtime.get_tile<fp32_t>(weight);
        w_meta = core::make_contiguous_torch_meta(weight->shape());
    }
    if (bias != nullptr)
    {
        b_ptr = &runtime.get_tile<fp32_t>(bias);
        b_meta = core::make_contiguous_torch_meta(bias->shape());
    }
    core::Tile<fp32_t> *gi_ptr = nullptr;
    core::Tile<fp32_t> *gw_ptr = nullptr;
    core::Tile<fp32_t> *gb_ptr = nullptr;
    core::TorchTileMeta gi_meta;
    core::TorchTileMeta gw_meta;
    core::TorchTileMeta gb_meta;
    if (need_grad_input && grad_input != nullptr)
    {
        gi_ptr = &runtime.get_tile<fp32_t>(grad_input);
        gi_meta = core::make_contiguous_torch_meta(
            grad_input->shape());
    }
    if (need_grad_weight && grad_weight != nullptr)
    {
        gw_ptr = &runtime.get_tile<fp32_t>(grad_weight);
        gw_meta = core::make_contiguous_torch_meta(
            grad_weight->shape());
    }
    if (need_grad_bias && grad_bias != nullptr)
    {
        gb_ptr = &runtime.get_tile<fp32_t>(grad_bias);
        gb_meta = core::make_contiguous_torch_meta(
            grad_bias->shape());
    }
    core::torch_layer_norm_backward_out(
        runtime.starpu_worker_hint(),
        go_t,
        go_meta,
        in_t,
        in_meta,
        mean_t,
        mean_meta,
        rstd_t,
        rstd_meta,
        w_ptr,
        w_ptr != nullptr ? &w_meta : nullptr,
        b_ptr,
        b_ptr != nullptr ? &b_meta : nullptr,
        gi_ptr,
        gi_ptr != nullptr ? &gi_meta : nullptr,
        gw_ptr,
        gw_ptr != nullptr ? &gw_meta : nullptr,
        gb_ptr,
        gb_ptr != nullptr ? &gb_meta : nullptr,
        normalized_ndim,
        need_grad_input,
        need_grad_weight,
        need_grad_bias);
}

void torch_embedding_dense_backward(
    TileGraph::TileNode *grad,
    TileGraph::TileNode *indices,
    TileGraph::TileNode *grad_weight)
{
    auto op = std::make_shared<TileTorchEmbeddingDenseBackwardOp>(
        grad,
        indices,
        grad_weight);
    grad->graph()->add_op(op);
}

void TileTorchEmbeddingDenseBackwardOp::execute(Runtime &runtime) const
{
    auto &g_t = runtime.get_tile<fp32_t>(grad);
    auto &idx_t = runtime.get_tile<int64_t>(indices);
    auto &gw_t = runtime.get_tile<fp32_t>(grad_weight);
    const core::TorchTileMeta g_meta =
        core::make_contiguous_torch_meta(grad->shape());
    const core::TorchTileMeta idx_meta =
        core::make_contiguous_torch_meta(indices->shape());
    const core::TorchTileMeta gw_meta =
        core::make_contiguous_torch_meta(grad_weight->shape());
    core::torch_embedding_dense_backward_out(
        runtime.starpu_worker_hint(),
        g_t,
        g_meta,
        idx_t,
        idx_meta,
        gw_t,
        gw_meta);
}

void torch_sdpa_backward(
    TileGraph::TileNode *q,
    TileGraph::TileNode *k,
    TileGraph::TileNode *v,
    TileGraph::TileNode *grad_out,
    TileGraph::TileNode *mask,
    TileGraph::TileNode *grad_q,
    TileGraph::TileNode *grad_k,
    TileGraph::TileNode *grad_v,
    bool is_causal,
    starpu::TorchDispatchArgs extra)
{
    auto op = std::make_shared<TileTorchSdpaBackwardOp>(
        q,
        k,
        v,
        grad_out,
        mask,
        grad_q,
        grad_k,
        grad_v,
        is_causal,
        extra);
    q->graph()->add_op(op);
}

void TileTorchSdpaBackwardOp::execute(Runtime &runtime) const
{
    auto &q_t = runtime.get_tile<fp32_t>(q);
    auto &k_t = runtime.get_tile<fp32_t>(k);
    auto &v_t = runtime.get_tile<fp32_t>(v);
    auto &go_t = runtime.get_tile<fp32_t>(grad_out);
    auto &gq_t = runtime.get_tile<fp32_t>(grad_q);
    auto &gk_t = runtime.get_tile<fp32_t>(grad_k);
    auto &gv_t = runtime.get_tile<fp32_t>(grad_v);
    // Prefer packed sizes/strides/offset (views into parent storage).
    const core::TorchTileMeta q_meta =
        core::meta_from_args_or_contiguous(
            extra, 0, false, q->shape());
    const core::TorchTileMeta k_meta =
        core::meta_from_args_or_contiguous(
            extra, 1, false, k->shape());
    const core::TorchTileMeta v_meta =
        core::meta_from_args_or_contiguous(
            extra, 2, false, v->shape());
    const core::TorchTileMeta go_meta =
        core::meta_from_args_or_contiguous(
            extra, 3, false, grad_out->shape());
    const core::TorchTileMeta gq_meta =
        core::meta_from_args_or_contiguous(
            extra, 0, true, grad_q->shape());
    const core::TorchTileMeta gk_meta =
        core::meta_from_args_or_contiguous(
            extra, 1, true, grad_k->shape());
    const core::TorchTileMeta gv_meta =
        core::meta_from_args_or_contiguous(
            extra, 2, true, grad_v->shape());
    core::Tile<bool_t> *mask_ptr = nullptr;
    core::TorchTileMeta mask_meta;
    if (mask != nullptr)
    {
        mask_ptr = &runtime.get_tile<bool_t>(mask);
        mask_meta = core::meta_from_args_or_contiguous(
            extra, 4, false, mask->shape());
    }
    core::torch_sdpa_backward_out(
        runtime.starpu_worker_hint(),
        q_t,
        q_meta,
        k_t,
        k_meta,
        v_t,
        v_meta,
        go_t,
        go_meta,
        mask_ptr,
        mask_ptr != nullptr ? &mask_meta : nullptr,
        gq_t,
        gq_meta,
        gk_t,
        gk_meta,
        gv_t,
        gv_meta,
        is_causal);
}

void torch_nll_loss_forward(
    TileGraph::TileNode *log_probs,
    TileGraph::TileNode *target,
    TileGraph::TileNode *loss,
    TileGraph::TileNode *total_weight,
    Index reduction,
    Index ignore_index,
    starpu::TorchDispatchArgs extra)
{
    auto op = std::make_shared<TileTorchNllLossForwardOp>(
        log_probs,
        target,
        loss,
        total_weight,
        reduction,
        ignore_index,
        extra);
    log_probs->graph()->add_op(op);
}

void TileTorchNllLossForwardOp::execute(Runtime &runtime) const
{
    auto &lp_t = runtime.get_tile<fp32_t>(log_probs);
    auto &tgt_t = runtime.get_tile<int64_t>(target);
    auto &loss_t = runtime.get_tile<fp32_t>(loss);
    auto &tw_t = runtime.get_tile<fp32_t>(total_weight);
    const core::TorchTileMeta lp_meta =
        core::meta_from_args_or_contiguous(
            extra,
            0,
            false,
            log_probs->shape());
    const core::TorchTileMeta tgt_meta =
        core::meta_from_args_or_contiguous(
            extra,
            1,
            false,
            target->shape());
    const core::TorchTileMeta loss_meta =
        core::meta_from_args_or_contiguous(
            extra,
            0,
            true,
            loss->shape());
    const core::TorchTileMeta tw_meta =
        core::meta_from_args_or_contiguous(
            extra,
            1,
            true,
            total_weight->shape());
    core::torch_nll_loss_forward_out(
        runtime.starpu_worker_hint(),
        lp_t,
        lp_meta,
        tgt_t,
        tgt_meta,
        loss_t,
        loss_meta,
        tw_t,
        tw_meta,
        reduction,
        ignore_index);
}

void torch_nll_loss_backward(
    TileGraph::TileNode *grad_output,
    TileGraph::TileNode *log_probs,
    TileGraph::TileNode *target,
    TileGraph::TileNode *total_weight,
    TileGraph::TileNode *grad_input,
    Index reduction,
    Index ignore_index,
    starpu::TorchDispatchArgs extra)
{
    auto op = std::make_shared<TileTorchNllLossBackwardOp>(
        grad_output,
        log_probs,
        target,
        total_weight,
        grad_input,
        reduction,
        ignore_index,
        extra);
    grad_output->graph()->add_op(op);
}

void TileTorchNllLossBackwardOp::execute(Runtime &runtime) const
{
    auto &go_t = runtime.get_tile<fp32_t>(grad_output);
    auto &lp_t = runtime.get_tile<fp32_t>(log_probs);
    auto &tgt_t = runtime.get_tile<int64_t>(target);
    auto &tw_t = runtime.get_tile<fp32_t>(total_weight);
    auto &gi_t = runtime.get_tile<fp32_t>(grad_input);
    const core::TorchTileMeta go_meta =
        core::meta_from_args_or_contiguous(
            extra,
            0,
            false,
            grad_output->shape());
    const core::TorchTileMeta lp_meta =
        core::meta_from_args_or_contiguous(
            extra,
            1,
            false,
            log_probs->shape());
    const core::TorchTileMeta tgt_meta =
        core::meta_from_args_or_contiguous(
            extra,
            2,
            false,
            target->shape());
    const core::TorchTileMeta tw_meta =
        core::meta_from_args_or_contiguous(
            extra,
            3,
            false,
            total_weight->shape());
    const core::TorchTileMeta gi_meta =
        core::meta_from_args_or_contiguous(
            extra,
            0,
            true,
            grad_input->shape());
    core::torch_nll_loss_backward_out(
        runtime.starpu_worker_hint(),
        go_t,
        go_meta,
        lp_t,
        lp_meta,
        tgt_t,
        tgt_meta,
        tw_t,
        tw_meta,
        gi_t,
        gi_meta,
        reduction,
        ignore_index);
}

} // namespace nntile::tile
