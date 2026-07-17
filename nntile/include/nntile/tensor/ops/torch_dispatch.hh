/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file include/nntile/tensor/ops/torch_dispatch.hh
 * TensorGraph ops for torch-native family codelets.
 *
 * @version 1.1.0
 */

#pragma once

#include <nntile/defs.h>

#ifndef NNTILE_TORCH_NATIVE_OPS
#error "nntile/tensor/ops/torch_dispatch.hh requires NNTILE_TORCH_NATIVE_OPS"
#endif

#include <string>
#include <vector>

#include <nntile/base_types.hh>
#include <nntile/starpu/torch_dispatch.hh>
#include <nntile/tensor/graph.hh>

namespace nntile
{
struct LoweringContext;
}

namespace nntile::tensor
{

struct TensorTorchUnaryOp : TensorGraph::OpNode
{
    starpu::TorchKind kind = starpu::TorchKind::Relu;
    starpu::TorchDispatchArgs extra{};
    TensorGraph::TensorNode *in = nullptr;
    TensorGraph::TensorNode *out = nullptr;

    TensorTorchUnaryOp() = default;
    TensorTorchUnaryOp(
        starpu::TorchKind kind_,
        TensorGraph::TensorNode *in_,
        TensorGraph::TensorNode *out_,
        starpu::TorchDispatchArgs extra_ = {}) :
        kind(kind_), extra(extra_), in(in_), out(out_)
    {
        inputs_ = {in};
        outputs_ = {out};
    }

    std::string op_name() const override
    {
        return "TORCH_UNARY";
    }

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorTorchUnaryOp>(*this);
    }

    void lower_to_tile(const LoweringContext &ctx) const override;
};

struct TensorTorchBinaryOp : TensorGraph::OpNode
{
    starpu::TorchKind kind = starpu::TorchKind::Mul;
    starpu::TorchDispatchArgs extra{};
    TensorGraph::TensorNode *a = nullptr;
    TensorGraph::TensorNode *b = nullptr;
    TensorGraph::TensorNode *out = nullptr;

    TensorTorchBinaryOp() = default;
    TensorTorchBinaryOp(
        starpu::TorchKind kind_,
        TensorGraph::TensorNode *a_,
        TensorGraph::TensorNode *b_,
        TensorGraph::TensorNode *out_,
        starpu::TorchDispatchArgs extra_ = {}) :
        kind(kind_), extra(extra_), a(a_), b(b_), out(out_)
    {
        inputs_ = {a, b};
        outputs_ = {out};
    }

    std::string op_name() const override
    {
        return "TORCH_BINARY";
    }

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorTorchBinaryOp>(*this);
    }

    void lower_to_tile(const LoweringContext &ctx) const override;
};

struct TensorTorchTernaryOp : TensorGraph::OpNode
{
    starpu::TorchKind kind = starpu::TorchKind::Addmm;
    starpu::TorchDispatchArgs extra{};
    TensorGraph::TensorNode *a = nullptr;
    TensorGraph::TensorNode *b = nullptr;
    TensorGraph::TensorNode *c = nullptr;
    TensorGraph::TensorNode *out = nullptr;

    TensorTorchTernaryOp() = default;
    TensorTorchTernaryOp(
        starpu::TorchKind kind_,
        TensorGraph::TensorNode *a_,
        TensorGraph::TensorNode *b_,
        TensorGraph::TensorNode *c_,
        TensorGraph::TensorNode *out_,
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
        return "TORCH_TERNARY";
    }

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorTorchTernaryOp>(*this);
    }

    void lower_to_tile(const LoweringContext &ctx) const override;
};

struct TensorTorchEmbeddingOp : TensorGraph::OpNode
{
    TensorGraph::TensorNode *weight = nullptr;
    TensorGraph::TensorNode *indices = nullptr;
    TensorGraph::TensorNode *out = nullptr;

    TensorTorchEmbeddingOp() = default;
    TensorTorchEmbeddingOp(
        TensorGraph::TensorNode *weight_,
        TensorGraph::TensorNode *indices_,
        TensorGraph::TensorNode *out_) :
        weight(weight_), indices(indices_), out(out_)
    {
        inputs_ = {weight, indices};
        outputs_ = {out};
    }

    std::string op_name() const override
    {
        return "TORCH_EMBEDDING";
    }

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorTorchEmbeddingOp>(*this);
    }

    void lower_to_tile(const LoweringContext &ctx) const override;
};

struct TensorTorchCatOp : TensorGraph::OpNode
{
    Index dim = 0;
    std::vector<TensorGraph::TensorNode *> inputs_tensors;
    TensorGraph::TensorNode *out = nullptr;

    TensorTorchCatOp() = default;
    TensorTorchCatOp(
        Index dim_,
        std::vector<TensorGraph::TensorNode *> ins,
        TensorGraph::TensorNode *out_) :
        dim(dim_), inputs_tensors(std::move(ins)), out(out_)
    {
        inputs_.assign(inputs_tensors.begin(), inputs_tensors.end());
        outputs_ = {out};
    }

    std::string op_name() const override
    {
        return "TORCH_CAT";
    }

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorTorchCatOp>(*this);
    }

    void lower_to_tile(const LoweringContext &ctx) const override;
};

TensorGraph::TensorNode *torch_unary(
    starpu::TorchKind kind,
    TensorGraph::TensorNode *in,
    const std::vector<Index> &out_shape,
    starpu::TorchDispatchArgs extra = {});

void torch_unary(
    starpu::TorchKind kind,
    TensorGraph::TensorNode *in,
    TensorGraph::TensorNode *out,
    starpu::TorchDispatchArgs extra = {});

TensorGraph::TensorNode *torch_binary(
    starpu::TorchKind kind,
    TensorGraph::TensorNode *a,
    TensorGraph::TensorNode *b,
    const std::vector<Index> &out_shape,
    starpu::TorchDispatchArgs extra = {});

void torch_binary(
    starpu::TorchKind kind,
    TensorGraph::TensorNode *a,
    TensorGraph::TensorNode *b,
    TensorGraph::TensorNode *out,
    starpu::TorchDispatchArgs extra = {});

TensorGraph::TensorNode *torch_ternary(
    starpu::TorchKind kind,
    TensorGraph::TensorNode *a,
    TensorGraph::TensorNode *b,
    TensorGraph::TensorNode *c,
    const std::vector<Index> &out_shape,
    starpu::TorchDispatchArgs extra = {});

void torch_ternary(
    starpu::TorchKind kind,
    TensorGraph::TensorNode *a,
    TensorGraph::TensorNode *b,
    TensorGraph::TensorNode *c,
    TensorGraph::TensorNode *out,
    starpu::TorchDispatchArgs extra = {});

TensorGraph::TensorNode *torch_embedding(
    TensorGraph::TensorNode *weight,
    TensorGraph::TensorNode *indices,
    const std::vector<Index> &out_shape);

TensorGraph::TensorNode *torch_cat(
    Index dim,
    const std::vector<TensorGraph::TensorNode *> &inputs,
    const std::vector<Index> &out_shape);

struct TensorTorchLayerNormOp : TensorGraph::OpNode
{
    Scalar eps = 1e-5;
    Index normalized_ndim = 1;
    TensorGraph::TensorNode *input = nullptr;
    TensorGraph::TensorNode *weight = nullptr;
    TensorGraph::TensorNode *bias = nullptr;
    TensorGraph::TensorNode *out = nullptr;
    TensorGraph::TensorNode *mean = nullptr;
    TensorGraph::TensorNode *rstd = nullptr;

    TensorTorchLayerNormOp() = default;
    TensorTorchLayerNormOp(
        TensorGraph::TensorNode *input_,
        TensorGraph::TensorNode *weight_,
        TensorGraph::TensorNode *bias_,
        TensorGraph::TensorNode *out_,
        TensorGraph::TensorNode *mean_,
        TensorGraph::TensorNode *rstd_,
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
        return "TORCH_NATIVE_LAYER_NORM";
    }

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorTorchLayerNormOp>(*this);
    }

    void lower_to_tile(const LoweringContext &ctx) const override;
};

void torch_layer_norm(
    TensorGraph::TensorNode *input,
    TensorGraph::TensorNode *weight,
    TensorGraph::TensorNode *bias,
    TensorGraph::TensorNode *out,
    TensorGraph::TensorNode *mean,
    TensorGraph::TensorNode *rstd,
    Index normalized_ndim,
    Scalar eps);

} // namespace nntile::tensor
