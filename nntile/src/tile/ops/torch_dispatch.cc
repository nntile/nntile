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
#include <nntile/dtype.hh>
#include <nntile/runtime.hh>
#include <nntile/starpu/handle.hh>

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
    TileGraph::TileNode *out,
    starpu::TorchDispatchArgs extra)
{
    if (weight == nullptr || indices == nullptr || out == nullptr)
    {
        throw std::invalid_argument("tile torch_embedding: null tile");
    }
    auto op = std::make_shared<TileTorchEmbeddingOp>(
        weight,
        indices,
        out,
        extra);
    weight->graph()->add_op(op);
}

void torch_where(
    TileGraph::TileNode *condition,
    TileGraph::TileNode *self,
    TileGraph::TileNode *other,
    TileGraph::TileNode *out,
    starpu::TorchDispatchArgs extra)
{
    if (condition == nullptr || self == nullptr ||
        other == nullptr || out == nullptr)
    {
        throw std::invalid_argument("tile torch_where: null tile");
    }
    auto op = std::make_shared<TileTorchWhereOp>(
        condition,
        self,
        other,
        out,
        extra);
    condition->graph()->add_op(op);
}

void torch_arange(
    TileGraph::TileNode *out,
    starpu::TorchDispatchArgs extra)
{
    if (out == nullptr)
    {
        throw std::invalid_argument("tile torch_arange: null tile");
    }
    auto op = std::make_shared<TileTorchArangeOp>(out, extra);
    out->graph()->add_op(op);
}

void torch_gt(
    TileGraph::TileNode *a,
    TileGraph::TileNode *b,
    TileGraph::TileNode *out,
    starpu::TorchDispatchArgs extra)
{
    if (a == nullptr || b == nullptr || out == nullptr)
    {
        throw std::invalid_argument("tile torch_gt: null tile");
    }
    auto op = std::make_shared<TileTorchGtOp>(a, b, out, extra);
    a->graph()->add_op(op);
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
    if (kind == starpu::TorchKind::Cast)
    {
        starpu::Handle in_h;
        starpu::Handle out_h;
        switch (in->dtype())
        {
        case DataType::FP32:
            in_h = runtime.get_tile<fp32_t>(in);
            break;
        case DataType::INT64:
            in_h = runtime.get_tile<int64_t>(in);
            break;
        case DataType::BOOL:
            in_h = runtime.get_tile<bool_t>(in);
            break;
        default:
            throw std::runtime_error(
                "TILE_TORCH_UNARY Cast: bad src dtype");
        }
        switch (out->dtype())
        {
        case DataType::FP32:
            out_h = runtime.get_tile<fp32_t>(out);
            break;
        case DataType::INT64:
            out_h = runtime.get_tile<int64_t>(out);
            break;
        case DataType::BOOL:
            out_h = runtime.get_tile<bool_t>(out);
            break;
        default:
            throw std::runtime_error(
                "TILE_TORCH_UNARY Cast: bad dst dtype");
        }
        core::torch_cast_out(
            runtime.starpu_worker_hint(),
            in_h,
            in_meta,
            out_h,
            out_meta,
            extra);
        return;
    }
    if (kind == starpu::TorchKind::Tril)
    {
        auto &in_t = runtime.get_tile<bool_t>(in);
        auto &out_t = runtime.get_tile<bool_t>(out);
        core::torch_unary_bool_out(
            runtime.starpu_worker_hint(),
            kind,
            in_t,
            in_meta,
            out_t,
            out_meta,
            extra);
        return;
    }
    if (in->dtype() == DataType::INT64)
    {
        auto &in_t = runtime.get_tile<int64_t>(in);
        auto &out_t = runtime.get_tile<int64_t>(out);
        core::torch_i64_unary_out(
            runtime.starpu_worker_hint(),
            kind,
            in_t,
            in_meta,
            out_t,
            out_meta,
            extra);
        return;
    }
    auto &in_t = runtime.get_tile<fp32_t>(in);
    auto &out_t = runtime.get_tile<fp32_t>(out);
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
    if (a->dtype() == DataType::INT64)
    {
        auto &a_t = runtime.get_tile<int64_t>(a);
        auto &b_t = runtime.get_tile<int64_t>(b);
        auto &out_t = runtime.get_tile<int64_t>(out);
        core::torch_i64_binary_out(
            runtime.starpu_worker_hint(),
            kind,
            a_t,
            a_meta,
            b_t,
            b_meta,
            out_t,
            out_meta,
            extra);
        return;
    }
    if (a->dtype() == DataType::FP32 && b->dtype() == DataType::BOOL &&
        out->dtype() == DataType::FP32)
    {
        auto &a_t = runtime.get_tile<fp32_t>(a);
        auto &b_t = runtime.get_tile<bool_t>(b);
        auto &out_t = runtime.get_tile<fp32_t>(out);
        core::torch_fp32_bool_mul_out(
            runtime.starpu_worker_hint(),
            a_t,
            a_meta,
            b_t,
            b_meta,
            out_t,
            out_meta,
            extra);
        return;
    }
    if (a->dtype() == DataType::BOOL)
    {
        auto &a_t = runtime.get_tile<bool_t>(a);
        auto &b_t = runtime.get_tile<bool_t>(b);
        auto &out_t = runtime.get_tile<bool_t>(out);
        core::torch_bool_binary_out(
            runtime.starpu_worker_hint(),
            kind,
            a_t,
            a_meta,
            b_t,
            b_meta,
            out_t,
            out_meta,
            extra);
        return;
    }
    auto &a_t = runtime.get_tile<fp32_t>(a);
    auto &b_t = runtime.get_tile<fp32_t>(b);
    auto &out_t = runtime.get_tile<fp32_t>(out);
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
    // Prefer packed sizes/strides/offset (sliced position_ids, etc.).
    const core::TorchTileMeta w_meta =
        core::meta_from_args_or_contiguous(
            extra, 0, false, weight->shape());
    const core::TorchTileMeta idx_meta =
        core::meta_from_args_or_contiguous(
            extra, 1, false, indices->shape());
    const core::TorchTileMeta out_meta =
        core::meta_from_args_or_contiguous(
            extra, 0, true, out->shape());
    core::torch_embedding_out(
        runtime.starpu_worker_hint(),
        w_t,
        w_meta,
        idx_t,
        idx_meta,
        out_t,
        out_meta);
}

void TileTorchWhereOp::execute(Runtime &runtime) const
{
    auto &cond_t = runtime.get_tile<bool_t>(condition);
    const core::TorchTileMeta cond_meta =
        core::meta_from_args_or_contiguous(
            extra, 0, false, condition->shape());
    const core::TorchTileMeta self_meta =
        core::meta_from_args_or_contiguous(
            extra, 1, false, self->shape());
    const core::TorchTileMeta other_meta =
        core::meta_from_args_or_contiguous(
            extra, 2, false, other->shape());
    const core::TorchTileMeta out_meta =
        core::meta_from_args_or_contiguous(
            extra, 0, true, out->shape());
    if (self->dtype() == DataType::INT64)
    {
        auto &self_t = runtime.get_tile<int64_t>(self);
        auto &other_t = runtime.get_tile<int64_t>(other);
        auto &out_t = runtime.get_tile<int64_t>(out);
        core::torch_where_i64_out(
            runtime.starpu_worker_hint(),
            cond_t,
            cond_meta,
            self_t,
            self_meta,
            other_t,
            other_meta,
            out_t,
            out_meta,
            extra);
        return;
    }
    auto &self_t = runtime.get_tile<fp32_t>(self);
    auto &other_t = runtime.get_tile<fp32_t>(other);
    auto &out_t = runtime.get_tile<fp32_t>(out);
    core::torch_where_out(
        runtime.starpu_worker_hint(),
        cond_t,
        cond_meta,
        self_t,
        self_meta,
        other_t,
        other_meta,
        out_t,
        out_meta);
}

void TileTorchArangeOp::execute(Runtime &runtime) const
{
    const core::TorchTileMeta out_meta =
        core::meta_from_args_or_contiguous(
            extra, 0, true, out->shape());
    if (out->dtype() == DataType::FP32)
    {
        auto &out_t = runtime.get_tile<fp32_t>(out);
        core::torch_arange_fp32_out(
            runtime.starpu_worker_hint(),
            out_t,
            out_meta,
            extra);
        return;
    }
    if (out->dtype() == DataType::BOOL)
    {
        auto &out_t = runtime.get_tile<bool_t>(out);
        core::torch_fill_bool_out(
            runtime.starpu_worker_hint(),
            out_t,
            out_meta,
            extra);
        return;
    }
    auto &out_t = runtime.get_tile<int64_t>(out);
    core::torch_arange_out(
        runtime.starpu_worker_hint(),
        out_t,
        out_meta,
        extra);
}

void TileTorchGtOp::execute(Runtime &runtime) const
{
    const core::TorchTileMeta a_meta =
        core::meta_from_args_or_contiguous(
            extra, 0, false, a->shape());
    const core::TorchTileMeta b_meta =
        core::meta_from_args_or_contiguous(
            extra, 1, false, b->shape());
    const core::TorchTileMeta out_meta =
        core::meta_from_args_or_contiguous(
            extra, 0, true, out->shape());
    auto &out_t = runtime.get_tile<bool_t>(out);
    if (a->dtype() == DataType::FP32)
    {
        auto &a_t = runtime.get_tile<fp32_t>(a);
        auto &b_t = runtime.get_tile<fp32_t>(b);
        core::torch_eq_fp32_out(
            runtime.starpu_worker_hint(),
            a_t,
            a_meta,
            b_t,
            b_meta,
            out_t,
            out_meta,
            extra);
        return;
    }
    auto &a_t = runtime.get_tile<int64_t>(a);
    auto &b_t = runtime.get_tile<int64_t>(b);
    core::torch_gt_out(
        runtime.starpu_worker_hint(),
        a_t,
        a_meta,
        b_t,
        b_meta,
        out_t,
        out_meta,
        extra);
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
    TileGraph::TileNode *grad_weight,
    starpu::TorchDispatchArgs extra)
{
    auto op = std::make_shared<TileTorchEmbeddingDenseBackwardOp>(
        grad,
        indices,
        grad_weight,
        extra);
    grad->graph()->add_op(op);
}

void TileTorchEmbeddingDenseBackwardOp::execute(Runtime &runtime) const
{
    auto &g_t = runtime.get_tile<fp32_t>(grad);
    auto &idx_t = runtime.get_tile<int64_t>(indices);
    auto &gw_t = runtime.get_tile<fp32_t>(grad_weight);
    const core::TorchTileMeta g_meta =
        core::meta_from_args_or_contiguous(
            extra, 0, false, grad->shape());
    const core::TorchTileMeta idx_meta =
        core::meta_from_args_or_contiguous(
            extra, 1, false, indices->shape());
    const core::TorchTileMeta gw_meta =
        core::meta_from_args_or_contiguous(
            extra, 0, true, grad_weight->shape());
    core::torch_embedding_dense_backward_out(
        runtime.starpu_worker_hint(),
        g_t,
        g_meta,
        idx_t,
        idx_meta,
        gw_t,
        gw_meta);
}

void torch_convolution(
    TileGraph::TileNode *input,
    TileGraph::TileNode *weight,
    TileGraph::TileNode *bias,
    TileGraph::TileNode *out,
    starpu::TorchDispatchArgs extra)
{
    auto op = std::make_shared<TileTorchConvolutionOp>(
        input,
        weight,
        bias,
        out,
        extra);
    input->graph()->add_op(op);
}

void TileTorchConvolutionOp::execute(Runtime &runtime) const
{
    auto &in_t = runtime.get_tile<fp32_t>(input);
    auto &w_t = runtime.get_tile<fp32_t>(weight);
    auto &out_t = runtime.get_tile<fp32_t>(out);
    const core::TorchTileMeta in_meta =
        core::meta_from_args_or_contiguous(extra, 0, false, input->shape());
    const core::TorchTileMeta w_meta =
        core::meta_from_args_or_contiguous(extra, 1, false, weight->shape());
    const core::TorchTileMeta out_meta =
        core::meta_from_args_or_contiguous(extra, 0, true, out->shape());
    core::Tile<fp32_t> *b_ptr = nullptr;
    core::TorchTileMeta b_meta;
    if (bias != nullptr)
    {
        b_ptr = &runtime.get_tile<fp32_t>(bias);
        b_meta = core::meta_from_args_or_contiguous(
            extra,
            2,
            false,
            bias->shape());
    }
    core::torch_convolution_out(
        runtime.starpu_worker_hint(),
        in_t,
        in_meta,
        w_t,
        w_meta,
        b_ptr,
        b_ptr != nullptr ? &b_meta : nullptr,
        out_t,
        out_meta,
        extra);
}

void torch_convolution_backward(
    TileGraph::TileNode *grad_out,
    TileGraph::TileNode *input,
    TileGraph::TileNode *weight,
    TileGraph::TileNode *grad_input,
    TileGraph::TileNode *grad_weight,
    TileGraph::TileNode *grad_bias,
    bool need_grad_input,
    bool need_grad_weight,
    bool need_grad_bias,
    starpu::TorchDispatchArgs extra)
{
    auto op = std::make_shared<TileTorchConvolutionBackwardOp>(
        grad_out,
        input,
        weight,
        grad_input,
        grad_weight,
        grad_bias,
        need_grad_input,
        need_grad_weight,
        need_grad_bias,
        extra);
    grad_out->graph()->add_op(op);
}

void TileTorchConvolutionBackwardOp::execute(Runtime &runtime) const
{
    auto &go_t = runtime.get_tile<fp32_t>(grad_out);
    auto &in_t = runtime.get_tile<fp32_t>(input);
    auto &w_t = runtime.get_tile<fp32_t>(weight);
    const core::TorchTileMeta go_meta =
        core::meta_from_args_or_contiguous(
            extra, 0, false, grad_out->shape());
    const core::TorchTileMeta in_meta =
        core::meta_from_args_or_contiguous(extra, 1, false, input->shape());
    const core::TorchTileMeta w_meta =
        core::meta_from_args_or_contiguous(extra, 2, false, weight->shape());
    core::Tile<fp32_t> *gi_ptr = nullptr;
    core::Tile<fp32_t> *gw_ptr = nullptr;
    core::Tile<fp32_t> *gb_ptr = nullptr;
    core::TorchTileMeta gi_meta;
    core::TorchTileMeta gw_meta;
    core::TorchTileMeta gb_meta;
    if (need_grad_input && grad_input != nullptr)
    {
        gi_ptr = &runtime.get_tile<fp32_t>(grad_input);
        gi_meta = core::meta_from_args_or_contiguous(
            extra, 0, true, grad_input->shape());
    }
    if (need_grad_weight && grad_weight != nullptr)
    {
        gw_ptr = &runtime.get_tile<fp32_t>(grad_weight);
        gw_meta = core::meta_from_args_or_contiguous(
            extra, 1, true, grad_weight->shape());
    }
    if (need_grad_bias && grad_bias != nullptr)
    {
        gb_ptr = &runtime.get_tile<fp32_t>(grad_bias);
        gb_meta = core::meta_from_args_or_contiguous(
            extra, 2, true, grad_bias->shape());
    }
    core::torch_convolution_backward_out(
        runtime.starpu_worker_hint(),
        go_t,
        go_meta,
        in_t,
        in_meta,
        w_t,
        w_meta,
        gi_ptr,
        gi_ptr != nullptr ? &gi_meta : nullptr,
        gw_ptr,
        gw_ptr != nullptr ? &gw_meta : nullptr,
        gb_ptr,
        gb_ptr != nullptr ? &gb_meta : nullptr,
        extra,
        need_grad_input,
        need_grad_weight,
        need_grad_bias);
}

void torch_max_pool2d_with_indices(
    TileGraph::TileNode *input,
    TileGraph::TileNode *out,
    TileGraph::TileNode *indices,
    starpu::TorchDispatchArgs extra)
{
    auto op = std::make_shared<TileTorchMaxPool2dWithIndicesOp>(
        input,
        out,
        indices,
        extra);
    input->graph()->add_op(op);
}

void TileTorchMaxPool2dWithIndicesOp::execute(Runtime &runtime) const
{
    auto &in_t = runtime.get_tile<fp32_t>(input);
    auto &out_t = runtime.get_tile<fp32_t>(out);
    auto &idx_t = runtime.get_tile<int64_t>(indices);
    const core::TorchTileMeta in_meta =
        core::meta_from_args_or_contiguous(extra, 0, false, input->shape());
    const core::TorchTileMeta out_meta =
        core::meta_from_args_or_contiguous(extra, 0, true, out->shape());
    const core::TorchTileMeta idx_meta =
        core::meta_from_args_or_contiguous(extra, 1, true, indices->shape());
    core::torch_max_pool2d_with_indices_out(
        runtime.starpu_worker_hint(),
        in_t,
        in_meta,
        out_t,
        out_meta,
        idx_t,
        idx_meta,
        extra);
}

void torch_max_pool2d_with_indices_backward(
    TileGraph::TileNode *grad_out,
    TileGraph::TileNode *input,
    TileGraph::TileNode *indices,
    TileGraph::TileNode *grad_input,
    starpu::TorchDispatchArgs extra)
{
    auto op = std::make_shared<TileTorchMaxPool2dWithIndicesBackwardOp>(
        grad_out,
        input,
        indices,
        grad_input,
        extra);
    grad_out->graph()->add_op(op);
}

void TileTorchMaxPool2dWithIndicesBackwardOp::execute(
    Runtime &runtime) const
{
    auto &go_t = runtime.get_tile<fp32_t>(grad_out);
    auto &in_t = runtime.get_tile<fp32_t>(input);
    auto &idx_t = runtime.get_tile<int64_t>(indices);
    auto &gi_t = runtime.get_tile<fp32_t>(grad_input);
    const core::TorchTileMeta go_meta =
        core::meta_from_args_or_contiguous(
            extra, 0, false, grad_out->shape());
    const core::TorchTileMeta in_meta =
        core::meta_from_args_or_contiguous(extra, 1, false, input->shape());
    const core::TorchTileMeta idx_meta =
        core::meta_from_args_or_contiguous(
            extra, 2, false, indices->shape());
    const core::TorchTileMeta gi_meta =
        core::meta_from_args_or_contiguous(
            extra, 0, true, grad_input->shape());
    core::torch_max_pool2d_with_indices_backward_out(
        runtime.starpu_worker_hint(),
        go_t,
        go_meta,
        in_t,
        in_meta,
        idx_t,
        idx_meta,
        gi_t,
        gi_meta,
        extra);
}

void torch_native_batch_norm(
    TileGraph::TileNode *input,
    TileGraph::TileNode *weight,
    TileGraph::TileNode *bias,
    TileGraph::TileNode *running_mean,
    TileGraph::TileNode *running_var,
    TileGraph::TileNode *out,
    TileGraph::TileNode *save_mean,
    TileGraph::TileNode *save_invstd,
    bool training,
    starpu::TorchDispatchArgs extra)
{
    auto op = std::make_shared<TileTorchNativeBatchNormOp>(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        out,
        save_mean,
        save_invstd,
        training,
        extra);
    input->graph()->add_op(op);
}

void TileTorchNativeBatchNormOp::execute(Runtime &runtime) const
{
    auto &in_t = runtime.get_tile<fp32_t>(input);
    auto &out_t = runtime.get_tile<fp32_t>(out);
    auto &sm_t = runtime.get_tile<fp32_t>(save_mean);
    auto &si_t = runtime.get_tile<fp32_t>(save_invstd);
    const core::TorchTileMeta in_meta =
        core::meta_from_args_or_contiguous(extra, 0, false, input->shape());
    const core::TorchTileMeta out_meta =
        core::meta_from_args_or_contiguous(extra, 0, true, out->shape());
    const core::TorchTileMeta sm_meta =
        core::meta_from_args_or_contiguous(
            extra, 1, true, save_mean->shape());
    const core::TorchTileMeta si_meta =
        core::meta_from_args_or_contiguous(
            extra, 2, true, save_invstd->shape());
    core::Tile<fp32_t> *w_ptr = nullptr;
    core::Tile<fp32_t> *b_ptr = nullptr;
    core::Tile<fp32_t> *rm_ptr = nullptr;
    core::Tile<fp32_t> *rv_ptr = nullptr;
    core::TorchTileMeta w_meta;
    core::TorchTileMeta b_meta;
    core::TorchTileMeta rm_meta;
    core::TorchTileMeta rv_meta;
    if (weight != nullptr)
    {
        w_ptr = &runtime.get_tile<fp32_t>(weight);
        w_meta = core::meta_from_args_or_contiguous(
            extra, 1, false, weight->shape());
    }
    if (bias != nullptr)
    {
        b_ptr = &runtime.get_tile<fp32_t>(bias);
        b_meta = core::meta_from_args_or_contiguous(
            extra, 2, false, bias->shape());
    }
    if (running_mean != nullptr)
    {
        rm_ptr = &runtime.get_tile<fp32_t>(running_mean);
        rm_meta = core::meta_from_args_or_contiguous(
            extra, 3, false, running_mean->shape());
    }
    if (running_var != nullptr)
    {
        rv_ptr = &runtime.get_tile<fp32_t>(running_var);
        rv_meta = core::meta_from_args_or_contiguous(
            extra, 4, false, running_var->shape());
    }
    core::torch_native_batch_norm_out(
        runtime.starpu_worker_hint(),
        in_t,
        in_meta,
        w_ptr,
        w_ptr != nullptr ? &w_meta : nullptr,
        b_ptr,
        b_ptr != nullptr ? &b_meta : nullptr,
        rm_ptr,
        rm_ptr != nullptr ? &rm_meta : nullptr,
        rv_ptr,
        rv_ptr != nullptr ? &rv_meta : nullptr,
        out_t,
        out_meta,
        sm_t,
        sm_meta,
        si_t,
        si_meta,
        extra,
        training);
}

void torch_native_batch_norm_backward(
    TileGraph::TileNode *grad_out,
    TileGraph::TileNode *input,
    TileGraph::TileNode *weight,
    TileGraph::TileNode *running_mean,
    TileGraph::TileNode *running_var,
    TileGraph::TileNode *save_mean,
    TileGraph::TileNode *save_invstd,
    TileGraph::TileNode *grad_input,
    TileGraph::TileNode *grad_weight,
    TileGraph::TileNode *grad_bias,
    bool need_grad_input,
    bool need_grad_weight,
    bool need_grad_bias,
    starpu::TorchDispatchArgs extra)
{
    auto op = std::make_shared<TileTorchNativeBatchNormBackwardOp>(
        grad_out,
        input,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_invstd,
        grad_input,
        grad_weight,
        grad_bias,
        need_grad_input,
        need_grad_weight,
        need_grad_bias,
        extra);
    grad_out->graph()->add_op(op);
}

void TileTorchNativeBatchNormBackwardOp::execute(Runtime &runtime) const
{
    auto &go_t = runtime.get_tile<fp32_t>(grad_out);
    auto &in_t = runtime.get_tile<fp32_t>(input);
    const core::TorchTileMeta go_meta =
        core::meta_from_args_or_contiguous(
            extra, 0, false, grad_out->shape());
    const core::TorchTileMeta in_meta =
        core::meta_from_args_or_contiguous(extra, 1, false, input->shape());
    core::Tile<fp32_t> *w_ptr = nullptr;
    core::Tile<fp32_t> *rm_ptr = nullptr;
    core::Tile<fp32_t> *rv_ptr = nullptr;
    core::Tile<fp32_t> *sm_ptr = nullptr;
    core::Tile<fp32_t> *si_ptr = nullptr;
    core::Tile<fp32_t> *gi_ptr = nullptr;
    core::Tile<fp32_t> *gw_ptr = nullptr;
    core::Tile<fp32_t> *gb_ptr = nullptr;
    core::TorchTileMeta w_meta;
    core::TorchTileMeta rm_meta;
    core::TorchTileMeta rv_meta;
    core::TorchTileMeta sm_meta;
    core::TorchTileMeta si_meta;
    core::TorchTileMeta gi_meta;
    core::TorchTileMeta gw_meta;
    core::TorchTileMeta gb_meta;
    if (weight != nullptr)
    {
        w_ptr = &runtime.get_tile<fp32_t>(weight);
        w_meta = core::meta_from_args_or_contiguous(
            extra, 2, false, weight->shape());
    }
    if (running_mean != nullptr)
    {
        rm_ptr = &runtime.get_tile<fp32_t>(running_mean);
        rm_meta = core::meta_from_args_or_contiguous(
            extra, 3, false, running_mean->shape());
    }
    if (running_var != nullptr)
    {
        rv_ptr = &runtime.get_tile<fp32_t>(running_var);
        rv_meta = core::meta_from_args_or_contiguous(
            extra, 4, false, running_var->shape());
    }
    if (save_mean != nullptr)
    {
        sm_ptr = &runtime.get_tile<fp32_t>(save_mean);
        sm_meta = core::meta_from_args_or_contiguous(
            extra, 5, false, save_mean->shape());
    }
    if (save_invstd != nullptr)
    {
        si_ptr = &runtime.get_tile<fp32_t>(save_invstd);
        si_meta = core::meta_from_args_or_contiguous(
            extra, 6, false, save_invstd->shape());
    }
    if (need_grad_input && grad_input != nullptr)
    {
        gi_ptr = &runtime.get_tile<fp32_t>(grad_input);
        gi_meta = core::meta_from_args_or_contiguous(
            extra, 0, true, grad_input->shape());
    }
    if (need_grad_weight && grad_weight != nullptr)
    {
        gw_ptr = &runtime.get_tile<fp32_t>(grad_weight);
        gw_meta = core::meta_from_args_or_contiguous(
            extra, 1, true, grad_weight->shape());
    }
    if (need_grad_bias && grad_bias != nullptr)
    {
        gb_ptr = &runtime.get_tile<fp32_t>(grad_bias);
        gb_meta = core::meta_from_args_or_contiguous(
            extra, 2, true, grad_bias->shape());
    }
    core::torch_native_batch_norm_backward_out(
        runtime.starpu_worker_hint(),
        go_t,
        go_meta,
        in_t,
        in_meta,
        w_ptr,
        w_ptr != nullptr ? &w_meta : nullptr,
        rm_ptr,
        rm_ptr != nullptr ? &rm_meta : nullptr,
        rv_ptr,
        rv_ptr != nullptr ? &rv_meta : nullptr,
        sm_ptr,
        sm_ptr != nullptr ? &sm_meta : nullptr,
        si_ptr,
        si_ptr != nullptr ? &si_meta : nullptr,
        gi_ptr,
        gi_ptr != nullptr ? &gi_meta : nullptr,
        gw_ptr,
        gw_ptr != nullptr ? &gw_meta : nullptr,
        gb_ptr,
        gb_ptr != nullptr ? &gb_meta : nullptr,
        extra,
        need_grad_input,
        need_grad_weight,
        need_grad_bias);
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
