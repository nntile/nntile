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

void torch_layer_norm_backward(
    TensorGraph::TensorNode *grad_out,
    TensorGraph::TensorNode *input,
    TensorGraph::TensorNode *mean,
    TensorGraph::TensorNode *rstd,
    TensorGraph::TensorNode *weight,
    TensorGraph::TensorNode *bias,
    TensorGraph::TensorNode *grad_input,
    TensorGraph::TensorNode *grad_weight,
    TensorGraph::TensorNode *grad_bias,
    Index normalized_ndim,
    bool need_grad_input,
    bool need_grad_weight,
    bool need_grad_bias)
{
    auto op = std::make_shared<TensorTorchLayerNormBackwardOp>(
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

void TensorTorchLayerNormBackwardOp::lower_to_tile(
    const LoweringContext &ctx) const
{
    const auto &vgo = tile_lower::tiles_of(ctx.tile_map, grad_out);
    const auto &vin = tile_lower::tiles_of(ctx.tile_map, input);
    const auto &vmean = tile_lower::tiles_of(ctx.tile_map, mean);
    const auto &vrstd = tile_lower::tiles_of(ctx.tile_map, rstd);
    require_single_tile("TORCH_LAYER_NORM_BWD", vgo);
    require_single_tile("TORCH_LAYER_NORM_BWD", vin);
    require_single_tile("TORCH_LAYER_NORM_BWD", vmean);
    require_single_tile("TORCH_LAYER_NORM_BWD", vrstd);
    TileGraph::TileNode *vw = nullptr;
    TileGraph::TileNode *vb = nullptr;
    TileGraph::TileNode *vgi = nullptr;
    TileGraph::TileNode *vgw = nullptr;
    TileGraph::TileNode *vgb = nullptr;
    if (weight != nullptr)
    {
        const auto &vt = tile_lower::tiles_of(ctx.tile_map, weight);
        require_single_tile("TORCH_LAYER_NORM_BWD", vt);
        vw = vt[0];
    }
    if (bias != nullptr)
    {
        const auto &vt = tile_lower::tiles_of(ctx.tile_map, bias);
        require_single_tile("TORCH_LAYER_NORM_BWD", vt);
        vb = vt[0];
    }
    if (need_grad_input && grad_input != nullptr)
    {
        const auto &vt = tile_lower::tiles_of(ctx.tile_map, grad_input);
        require_single_tile("TORCH_LAYER_NORM_BWD", vt);
        vgi = vt[0];
    }
    if (need_grad_weight && grad_weight != nullptr)
    {
        const auto &vt =
            tile_lower::tiles_of(ctx.tile_map, grad_weight);
        require_single_tile("TORCH_LAYER_NORM_BWD", vt);
        vgw = vt[0];
    }
    if (need_grad_bias && grad_bias != nullptr)
    {
        const auto &vt = tile_lower::tiles_of(ctx.tile_map, grad_bias);
        require_single_tile("TORCH_LAYER_NORM_BWD", vt);
        vgb = vt[0];
    }
    tile::torch_layer_norm_backward(
        vgo[0],
        vin[0],
        vmean[0],
        vrstd[0],
        vw,
        vb,
        vgi,
        vgw,
        vgb,
        normalized_ndim,
        need_grad_input,
        need_grad_weight,
        need_grad_bias);
}

void torch_embedding_dense_backward(
    TensorGraph::TensorNode *grad,
    TensorGraph::TensorNode *indices,
    TensorGraph::TensorNode *grad_weight)
{
    auto op = std::make_shared<TensorTorchEmbeddingDenseBackwardOp>(
        grad,
        indices,
        grad_weight);
    grad->graph()->add_op(op);
}

void TensorTorchEmbeddingDenseBackwardOp::lower_to_tile(
    const LoweringContext &ctx) const
{
    const auto &vg = tile_lower::tiles_of(ctx.tile_map, grad);
    const auto &vi = tile_lower::tiles_of(ctx.tile_map, indices);
    const auto &vgw = tile_lower::tiles_of(ctx.tile_map, grad_weight);
    require_single_tile("TORCH_EMBEDDING_DENSE_BWD", vg);
    require_single_tile("TORCH_EMBEDDING_DENSE_BWD", vi);
    require_single_tile("TORCH_EMBEDDING_DENSE_BWD", vgw);
    tile::torch_embedding_dense_backward(vg[0], vi[0], vgw[0]);
}

void torch_sdpa_backward(
    TensorGraph::TensorNode *q,
    TensorGraph::TensorNode *k,
    TensorGraph::TensorNode *v,
    TensorGraph::TensorNode *grad_out,
    TensorGraph::TensorNode *mask,
    TensorGraph::TensorNode *grad_q,
    TensorGraph::TensorNode *grad_k,
    TensorGraph::TensorNode *grad_v,
    bool is_causal,
    starpu::TorchDispatchArgs extra)
{
    auto op = std::make_shared<TensorTorchSdpaBackwardOp>(
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

void TensorTorchSdpaBackwardOp::lower_to_tile(
    const LoweringContext &ctx) const
{
    const auto &vq = tile_lower::tiles_of(ctx.tile_map, q);
    const auto &vk = tile_lower::tiles_of(ctx.tile_map, k);
    const auto &vv = tile_lower::tiles_of(ctx.tile_map, v);
    const auto &vgo = tile_lower::tiles_of(ctx.tile_map, grad_out);
    const auto &vgq = tile_lower::tiles_of(ctx.tile_map, grad_q);
    const auto &vgk = tile_lower::tiles_of(ctx.tile_map, grad_k);
    const auto &vgv = tile_lower::tiles_of(ctx.tile_map, grad_v);
    require_single_tile("TORCH_SDPA_BWD", vq);
    require_single_tile("TORCH_SDPA_BWD", vk);
    require_single_tile("TORCH_SDPA_BWD", vv);
    require_single_tile("TORCH_SDPA_BWD", vgo);
    require_single_tile("TORCH_SDPA_BWD", vgq);
    require_single_tile("TORCH_SDPA_BWD", vgk);
    require_single_tile("TORCH_SDPA_BWD", vgv);
    TileGraph::TileNode *vm = nullptr;
    if (mask != nullptr)
    {
        const auto &vt = tile_lower::tiles_of(ctx.tile_map, mask);
        require_single_tile("TORCH_SDPA_BWD", vt);
        vm = vt[0];
    }
    tile::torch_sdpa_backward(
        vq[0],
        vk[0],
        vv[0],
        vgo[0],
        vm,
        vgq[0],
        vgk[0],
        vgv[0],
        is_causal,
        extra);
}

void torch_nll_loss_forward(
    TensorGraph::TensorNode *log_probs,
    TensorGraph::TensorNode *target,
    TensorGraph::TensorNode *loss,
    TensorGraph::TensorNode *total_weight,
    Index reduction,
    Index ignore_index,
    starpu::TorchDispatchArgs extra)
{
    auto op = std::make_shared<TensorTorchNllLossForwardOp>(
        log_probs,
        target,
        loss,
        total_weight,
        reduction,
        ignore_index,
        extra);
    log_probs->graph()->add_op(op);
}

void TensorTorchNllLossForwardOp::lower_to_tile(
    const LoweringContext &ctx) const
{
    const auto &vlp = tile_lower::tiles_of(ctx.tile_map, log_probs);
    const auto &vtgt = tile_lower::tiles_of(ctx.tile_map, target);
    const auto &vloss = tile_lower::tiles_of(ctx.tile_map, loss);
    const auto &vtw = tile_lower::tiles_of(ctx.tile_map, total_weight);
    require_single_tile("TORCH_NLL_FWD", vlp);
    require_single_tile("TORCH_NLL_FWD", vtgt);
    require_single_tile("TORCH_NLL_FWD", vloss);
    require_single_tile("TORCH_NLL_FWD", vtw);
    tile::torch_nll_loss_forward(
        vlp[0],
        vtgt[0],
        vloss[0],
        vtw[0],
        reduction,
        ignore_index,
        extra);
}

void torch_nll_loss_backward(
    TensorGraph::TensorNode *grad_output,
    TensorGraph::TensorNode *log_probs,
    TensorGraph::TensorNode *target,
    TensorGraph::TensorNode *total_weight,
    TensorGraph::TensorNode *grad_input,
    Index reduction,
    Index ignore_index,
    starpu::TorchDispatchArgs extra)
{
    auto op = std::make_shared<TensorTorchNllLossBackwardOp>(
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

void TensorTorchNllLossBackwardOp::lower_to_tile(
    const LoweringContext &ctx) const
{
    const auto &vgo = tile_lower::tiles_of(ctx.tile_map, grad_output);
    const auto &vlp = tile_lower::tiles_of(ctx.tile_map, log_probs);
    const auto &vtgt = tile_lower::tiles_of(ctx.tile_map, target);
    const auto &vtw = tile_lower::tiles_of(ctx.tile_map, total_weight);
    const auto &vgi = tile_lower::tiles_of(ctx.tile_map, grad_input);
    require_single_tile("TORCH_NLL_BWD", vgo);
    require_single_tile("TORCH_NLL_BWD", vlp);
    require_single_tile("TORCH_NLL_BWD", vtgt);
    require_single_tile("TORCH_NLL_BWD", vtw);
    require_single_tile("TORCH_NLL_BWD", vgi);
    tile::torch_nll_loss_backward(
        vgo[0],
        vlp[0],
        vtgt[0],
        vtw[0],
        vgi[0],
        reduction,
        ignore_index,
        extra);
}

} // namespace nntile::tensor
