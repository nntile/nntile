/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file include/nntile/tile/ops/torch_dispatch.hh
 * TileGraph ops for torch-native family codelets.
 *
 * @version 1.1.0
 */

#pragma once

#include <nntile/defs.h>

#ifndef NNTILE_TORCH_NATIVE_OPS
#error "nntile/tile/ops/torch_dispatch.hh requires NNTILE_TORCH_NATIVE_OPS"
#endif

#include <string>
#include <vector>

#include <nntile/base_types.hh>
#include <nntile/starpu/torch_dispatch.hh>
#include <nntile/tile/graph.hh>

namespace nntile::tile
{

struct TileTorchUnaryOp : TileGraph::OpNode
{
    starpu::TorchKind kind = starpu::TorchKind::Relu;
    starpu::TorchDispatchArgs extra{};
    TileGraph::TileNode *in = nullptr;
    TileGraph::TileNode *out = nullptr;

    TileTorchUnaryOp() = default;
    TileTorchUnaryOp(
        starpu::TorchKind kind_,
        TileGraph::TileNode *in_,
        TileGraph::TileNode *out_,
        starpu::TorchDispatchArgs extra_ = {}) :
        kind(kind_), extra(extra_), in(in_), out(out_)
    {
        inputs_ = {in};
        outputs_ = {out};
    }

    std::string op_name() const override
    {
        return "TILE_TORCH_UNARY";
    }

    void execute(Runtime &runtime) const override;

    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileTorchUnaryOp>(*this);
    }
};

struct TileTorchBinaryOp : TileGraph::OpNode
{
    starpu::TorchKind kind = starpu::TorchKind::Mul;
    starpu::TorchDispatchArgs extra{};
    TileGraph::TileNode *a = nullptr;
    TileGraph::TileNode *b = nullptr;
    TileGraph::TileNode *out = nullptr;

    TileTorchBinaryOp() = default;
    TileTorchBinaryOp(
        starpu::TorchKind kind_,
        TileGraph::TileNode *a_,
        TileGraph::TileNode *b_,
        TileGraph::TileNode *out_,
        starpu::TorchDispatchArgs extra_ = {}) :
        kind(kind_), extra(extra_), a(a_), b(b_), out(out_)
    {
        inputs_ = {a, b};
        outputs_ = {out};
    }

    std::string op_name() const override
    {
        return "TILE_TORCH_BINARY";
    }

    void execute(Runtime &runtime) const override;

    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileTorchBinaryOp>(*this);
    }
};

struct TileTorchTernaryOp : TileGraph::OpNode
{
    starpu::TorchKind kind = starpu::TorchKind::Addmm;
    starpu::TorchDispatchArgs extra{};
    TileGraph::TileNode *a = nullptr;
    TileGraph::TileNode *b = nullptr;
    TileGraph::TileNode *c = nullptr;
    TileGraph::TileNode *out = nullptr;

    TileTorchTernaryOp() = default;
    TileTorchTernaryOp(
        starpu::TorchKind kind_,
        TileGraph::TileNode *a_,
        TileGraph::TileNode *b_,
        TileGraph::TileNode *c_,
        TileGraph::TileNode *out_,
        starpu::TorchDispatchArgs extra_ = {}) :
        kind(kind_),
        extra(extra_),
        a(a_),
        b(b_),
        c(c_),
        out(out_)
    {
        inputs_ = {a, b, c};
        outputs_ = {out};
    }

    std::string op_name() const override
    {
        return "TILE_TORCH_TERNARY";
    }

    void execute(Runtime &runtime) const override;

    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileTorchTernaryOp>(*this);
    }
};

struct TileTorchEmbeddingOp : TileGraph::OpNode
{
    TileGraph::TileNode *weight = nullptr;
    TileGraph::TileNode *indices = nullptr;
    TileGraph::TileNode *out = nullptr;

    TileTorchEmbeddingOp() = default;
    TileTorchEmbeddingOp(
        TileGraph::TileNode *weight_,
        TileGraph::TileNode *indices_,
        TileGraph::TileNode *out_) :
        weight(weight_), indices(indices_), out(out_)
    {
        inputs_ = {weight, indices};
        outputs_ = {out};
    }

    std::string op_name() const override
    {
        return "TILE_TORCH_EMBEDDING";
    }

    void execute(Runtime &runtime) const override;

    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileTorchEmbeddingOp>(*this);
    }
};

struct TileTorchCatOp : TileGraph::OpNode
{
    Index dim = 0;
    std::vector<TileGraph::TileNode *> inputs_tiles;
    TileGraph::TileNode *out = nullptr;

    TileTorchCatOp() = default;
    TileTorchCatOp(
        Index dim_,
        std::vector<TileGraph::TileNode *> ins,
        TileGraph::TileNode *out_) :
        dim(dim_), inputs_tiles(std::move(ins)), out(out_)
    {
        inputs_.assign(inputs_tiles.begin(), inputs_tiles.end());
        outputs_ = {out};
    }

    std::string op_name() const override
    {
        return "TILE_TORCH_CAT";
    }

    void execute(Runtime &runtime) const override;

    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileTorchCatOp>(*this);
    }
};

void torch_unary(
    starpu::TorchKind kind,
    TileGraph::TileNode *in,
    TileGraph::TileNode *out,
    starpu::TorchDispatchArgs extra = {});

void torch_binary(
    starpu::TorchKind kind,
    TileGraph::TileNode *a,
    TileGraph::TileNode *b,
    TileGraph::TileNode *out,
    starpu::TorchDispatchArgs extra = {});

void torch_ternary(
    starpu::TorchKind kind,
    TileGraph::TileNode *a,
    TileGraph::TileNode *b,
    TileGraph::TileNode *c,
    TileGraph::TileNode *out,
    starpu::TorchDispatchArgs extra = {});

void torch_embedding(
    TileGraph::TileNode *weight,
    TileGraph::TileNode *indices,
    TileGraph::TileNode *out);

void torch_cat(
    Index dim,
    const std::vector<TileGraph::TileNode *> &inputs,
    TileGraph::TileNode *out);

struct TileTorchLayerNormOp : TileGraph::OpNode
{
    Scalar eps = 1e-5;
    Index normalized_ndim = 1;
    TileGraph::TileNode *input = nullptr;
    TileGraph::TileNode *weight = nullptr;
    TileGraph::TileNode *bias = nullptr;
    TileGraph::TileNode *out = nullptr;
    TileGraph::TileNode *mean = nullptr;
    TileGraph::TileNode *rstd = nullptr;

    TileTorchLayerNormOp() = default;
    TileTorchLayerNormOp(
        TileGraph::TileNode *input_,
        TileGraph::TileNode *weight_,
        TileGraph::TileNode *bias_,
        TileGraph::TileNode *out_,
        TileGraph::TileNode *mean_,
        TileGraph::TileNode *rstd_,
        Index normalized_ndim_,
        Scalar eps_) :
        eps(eps_),
        normalized_ndim(normalized_ndim_),
        input(input_),
        weight(weight_),
        bias(bias_),
        out(out_),
        mean(mean_),
        rstd(rstd_)
    {
        inputs_ = {input};
        if (weight != nullptr)
        {
            inputs_.push_back(weight);
        }
        if (bias != nullptr)
        {
            inputs_.push_back(bias);
        }
        outputs_ = {out, mean, rstd};
    }

    std::string op_name() const override
    {
        return "TILE_TORCH_LAYER_NORM";
    }

    void execute(Runtime &runtime) const override;

    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileTorchLayerNormOp>(*this);
    }
};

void torch_layer_norm(
    TileGraph::TileNode *input,
    TileGraph::TileNode *weight,
    TileGraph::TileNode *bias,
    TileGraph::TileNode *out,
    TileGraph::TileNode *mean,
    TileGraph::TileNode *rstd,
    Index normalized_ndim,
    Scalar eps);

} // namespace nntile::tile
