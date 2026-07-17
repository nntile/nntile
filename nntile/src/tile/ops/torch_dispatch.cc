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
        core::make_contiguous_torch_meta(in->shape());
    const core::TorchTileMeta out_meta =
        core::make_contiguous_torch_meta(out->shape());
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
        core::make_contiguous_torch_meta(a->shape());
    const core::TorchTileMeta b_meta =
        core::make_contiguous_torch_meta(b->shape());
    const core::TorchTileMeta out_meta =
        core::make_contiguous_torch_meta(out->shape());
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
        core::make_contiguous_torch_meta(a->shape());
    const core::TorchTileMeta b_meta =
        core::make_contiguous_torch_meta(b->shape());
    const core::TorchTileMeta c_meta =
        core::make_contiguous_torch_meta(c->shape());
    const core::TorchTileMeta out_meta =
        core::make_contiguous_torch_meta(out->shape());
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

} // namespace nntile::tile
