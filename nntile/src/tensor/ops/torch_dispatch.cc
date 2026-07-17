/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file src/tensor/ops/torch_dispatch.cc
 * TensorGraph torch-native family ops (single-tile lower).
 *
 * @version 1.1.0
 */

#include "nntile/tensor/ops/torch_dispatch.hh"

#include <stdexcept>
#include <utility>

#include <nntile/base_types.hh>
#include <nntile/tensor/tile_lowering_helpers.hh>
#include <nntile/tile/ops/torch_dispatch.hh>

namespace nntile::tensor
{

namespace
{

void require_single_tile(
    const char *op,
    const std::vector<TileGraph::TileNode *> &tiles)
{
    if (tiles.size() != 1)
    {
        throw std::runtime_error(
            std::string("lower_to_tile ") + op +
            ": requires exactly one tile per operand "
            "(untiled tensors only)");
    }
}

} // namespace

TensorGraph::TensorNode *torch_unary(
    starpu::TorchKind kind,
    TensorGraph::TensorNode *in,
    const std::vector<Index> &out_shape,
    starpu::TorchDispatchArgs extra)
{
    if (in == nullptr)
    {
        throw std::invalid_argument("torch_unary: null input");
    }
    TensorGraph::TensorNode *out =
        in->graph()->emplace_data(out_shape, in->dtype());
    torch_unary(kind, in, out, extra);
    return out;
}

void torch_unary(
    starpu::TorchKind kind,
    TensorGraph::TensorNode *in,
    TensorGraph::TensorNode *out,
    starpu::TorchDispatchArgs extra)
{
    if (in == nullptr || out == nullptr)
    {
        throw std::invalid_argument("torch_unary: null tensor");
    }
    if (in->graph() != out->graph())
    {
        throw std::invalid_argument("torch_unary: graph mismatch");
    }
    auto op = std::make_shared<TensorTorchUnaryOp>(
        kind,
        in,
        out,
        extra);
    in->graph()->add_op(op);
}

TensorGraph::TensorNode *torch_binary(
    starpu::TorchKind kind,
    TensorGraph::TensorNode *a,
    TensorGraph::TensorNode *b,
    const std::vector<Index> &out_shape,
    starpu::TorchDispatchArgs extra)
{
    if (a == nullptr || b == nullptr)
    {
        throw std::invalid_argument("torch_binary: null input");
    }
    if (a->graph() != b->graph())
    {
        throw std::invalid_argument("torch_binary: graph mismatch");
    }
    TensorGraph::TensorNode *out =
        a->graph()->emplace_data(out_shape, a->dtype());
    torch_binary(kind, a, b, out, extra);
    return out;
}

void torch_binary(
    starpu::TorchKind kind,
    TensorGraph::TensorNode *a,
    TensorGraph::TensorNode *b,
    TensorGraph::TensorNode *out,
    starpu::TorchDispatchArgs extra)
{
    if (a == nullptr || b == nullptr || out == nullptr)
    {
        throw std::invalid_argument("torch_binary: null tensor");
    }
    auto op = std::make_shared<TensorTorchBinaryOp>(
        kind,
        a,
        b,
        out,
        extra);
    a->graph()->add_op(op);
}

TensorGraph::TensorNode *torch_ternary(
    starpu::TorchKind kind,
    TensorGraph::TensorNode *a,
    TensorGraph::TensorNode *b,
    TensorGraph::TensorNode *c,
    const std::vector<Index> &out_shape,
    starpu::TorchDispatchArgs extra)
{
    TensorGraph::TensorNode *out =
        a->graph()->emplace_data(out_shape, a->dtype());
    torch_ternary(kind, a, b, c, out, extra);
    return out;
}

void torch_ternary(
    starpu::TorchKind kind,
    TensorGraph::TensorNode *a,
    TensorGraph::TensorNode *b,
    TensorGraph::TensorNode *c,
    TensorGraph::TensorNode *out,
    starpu::TorchDispatchArgs extra)
{
    auto op = std::make_shared<TensorTorchTernaryOp>(
        kind,
        a,
        b,
        c,
        out,
        extra);
    a->graph()->add_op(op);
}

TensorGraph::TensorNode *torch_embedding(
    TensorGraph::TensorNode *weight,
    TensorGraph::TensorNode *indices,
    const std::vector<Index> &out_shape)
{
    TensorGraph::TensorNode *out =
        weight->graph()->emplace_data(out_shape, weight->dtype());
    auto op = std::make_shared<TensorTorchEmbeddingOp>(
        weight,
        indices,
        out);
    weight->graph()->add_op(op);
    return out;
}

TensorGraph::TensorNode *torch_cat(
    Index dim,
    const std::vector<TensorGraph::TensorNode *> &inputs,
    const std::vector<Index> &out_shape)
{
    if (inputs.empty())
    {
        throw std::invalid_argument("torch_cat: empty inputs");
    }
    TensorGraph::TensorNode *out =
        inputs[0]->graph()->emplace_data(
            out_shape,
            inputs[0]->dtype());
    auto op = std::make_shared<TensorTorchCatOp>(dim, inputs, out);
    inputs[0]->graph()->add_op(op);
    return out;
}

void TensorTorchUnaryOp::lower_to_tile(const LoweringContext &ctx) const
{
    const auto &vin = tile_lower::tiles_of(ctx.tile_map, in);
    const auto &vout = tile_lower::tiles_of(ctx.tile_map, out);
    require_single_tile("TORCH_UNARY", vin);
    require_single_tile("TORCH_UNARY", vout);
    tile::torch_unary(kind, vin[0], vout[0], extra);
}

void TensorTorchBinaryOp::lower_to_tile(const LoweringContext &ctx) const
{
    const auto &va = tile_lower::tiles_of(ctx.tile_map, a);
    const auto &vb = tile_lower::tiles_of(ctx.tile_map, b);
    const auto &vout = tile_lower::tiles_of(ctx.tile_map, out);
    require_single_tile("TORCH_BINARY", va);
    require_single_tile("TORCH_BINARY", vb);
    require_single_tile("TORCH_BINARY", vout);
    tile::torch_binary(kind, va[0], vb[0], vout[0], extra);
}

void TensorTorchTernaryOp::lower_to_tile(const LoweringContext &ctx) const
{
    const auto &va = tile_lower::tiles_of(ctx.tile_map, a);
    const auto &vb = tile_lower::tiles_of(ctx.tile_map, b);
    const auto &vc = tile_lower::tiles_of(ctx.tile_map, c);
    const auto &vout = tile_lower::tiles_of(ctx.tile_map, out);
    require_single_tile("TORCH_TERNARY", va);
    require_single_tile("TORCH_TERNARY", vb);
    require_single_tile("TORCH_TERNARY", vc);
    require_single_tile("TORCH_TERNARY", vout);
    tile::torch_ternary(kind, va[0], vb[0], vc[0], vout[0], extra);
}

void TensorTorchEmbeddingOp::lower_to_tile(const LoweringContext &ctx) const
{
    const auto &vw = tile_lower::tiles_of(ctx.tile_map, weight);
    const auto &vi = tile_lower::tiles_of(ctx.tile_map, indices);
    const auto &vout = tile_lower::tiles_of(ctx.tile_map, out);
    require_single_tile("TORCH_EMBEDDING", vw);
    require_single_tile("TORCH_EMBEDDING", vi);
    require_single_tile("TORCH_EMBEDDING", vout);
    tile::torch_embedding(vw[0], vi[0], vout[0]);
}

void TensorTorchCatOp::lower_to_tile(const LoweringContext &ctx) const
{
    std::vector<TileGraph::TileNode *> tiles;
    tiles.reserve(inputs_tensors.size());
    for (auto *t : inputs_tensors)
    {
        const auto &vt = tile_lower::tiles_of(ctx.tile_map, t);
        require_single_tile("TORCH_CAT", vt);
        tiles.push_back(vt[0]);
    }
    const auto &vout = tile_lower::tiles_of(ctx.tile_map, out);
    require_single_tile("TORCH_CAT", vout);
    tile::torch_cat(dim, tiles, vout[0]);
}

void torch_layer_norm(
    TensorGraph::TensorNode *input,
    TensorGraph::TensorNode *weight,
    TensorGraph::TensorNode *bias,
    TensorGraph::TensorNode *out,
    TensorGraph::TensorNode *mean,
    TensorGraph::TensorNode *rstd,
    Index normalized_ndim,
    Scalar eps)
{
    auto op = std::make_shared<TensorTorchLayerNormOp>(
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

void TensorTorchLayerNormOp::lower_to_tile(const LoweringContext &ctx) const
{
    const auto &vin = tile_lower::tiles_of(ctx.tile_map, input);
    const auto &vout = tile_lower::tiles_of(ctx.tile_map, out);
    const auto &vmean = tile_lower::tiles_of(ctx.tile_map, mean);
    const auto &vrstd = tile_lower::tiles_of(ctx.tile_map, rstd);
    require_single_tile("TORCH_LAYER_NORM", vin);
    require_single_tile("TORCH_LAYER_NORM", vout);
    require_single_tile("TORCH_LAYER_NORM", vmean);
    require_single_tile("TORCH_LAYER_NORM", vrstd);
    TileGraph::TileNode *vw = nullptr;
    TileGraph::TileNode *vb = nullptr;
    if (weight != nullptr)
    {
        const auto &vt = tile_lower::tiles_of(ctx.tile_map, weight);
        require_single_tile("TORCH_LAYER_NORM", vt);
        vw = vt[0];
    }
    if (bias != nullptr)
    {
        const auto &vt = tile_lower::tiles_of(ctx.tile_map, bias);
        require_single_tile("TORCH_LAYER_NORM", vt);
        vb = vt[0];
    }
    tile::torch_layer_norm(
        vin[0],
        vw,
        vb,
        vout[0],
        vmean[0],
        vrstd[0],
        normalized_ndim,
        eps);
}

} // namespace nntile::tensor
