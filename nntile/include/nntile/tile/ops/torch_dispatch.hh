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

struct TileTorchLayerNormBackwardOp : TileGraph::OpNode
{
    Index normalized_ndim = 1;
    bool need_grad_input = false;
    bool need_grad_weight = false;
    bool need_grad_bias = false;
    TileGraph::TileNode *grad_out = nullptr;
    TileGraph::TileNode *input = nullptr;
    TileGraph::TileNode *mean = nullptr;
    TileGraph::TileNode *rstd = nullptr;
    TileGraph::TileNode *weight = nullptr;
    TileGraph::TileNode *bias = nullptr;
    TileGraph::TileNode *grad_input = nullptr;
    TileGraph::TileNode *grad_weight = nullptr;
    TileGraph::TileNode *grad_bias = nullptr;

    TileTorchLayerNormBackwardOp() = default;
    TileTorchLayerNormBackwardOp(
        TileGraph::TileNode *grad_out_,
        TileGraph::TileNode *input_,
        TileGraph::TileNode *mean_,
        TileGraph::TileNode *rstd_,
        TileGraph::TileNode *weight_,
        TileGraph::TileNode *bias_,
        TileGraph::TileNode *grad_input_,
        TileGraph::TileNode *grad_weight_,
        TileGraph::TileNode *grad_bias_,
        Index normalized_ndim_,
        bool need_grad_input_,
        bool need_grad_weight_,
        bool need_grad_bias_) :
        normalized_ndim(normalized_ndim_),
        need_grad_input(need_grad_input_),
        need_grad_weight(need_grad_weight_),
        need_grad_bias(need_grad_bias_),
        grad_out(grad_out_),
        input(input_),
        mean(mean_),
        rstd(rstd_),
        weight(weight_),
        bias(bias_),
        grad_input(grad_input_),
        grad_weight(grad_weight_),
        grad_bias(grad_bias_)
    {
        inputs_ = {grad_out, input, mean, rstd};
        if (weight != nullptr)
        {
            inputs_.push_back(weight);
        }
        if (bias != nullptr)
        {
            inputs_.push_back(bias);
        }
        if (need_grad_input && grad_input != nullptr)
        {
            outputs_.push_back(grad_input);
        }
        if (need_grad_weight && grad_weight != nullptr)
        {
            outputs_.push_back(grad_weight);
        }
        if (need_grad_bias && grad_bias != nullptr)
        {
            outputs_.push_back(grad_bias);
        }
    }

    std::string op_name() const override
    {
        return "TILE_TORCH_LAYER_NORM_BACKWARD";
    }

    void execute(Runtime &runtime) const override;

    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileTorchLayerNormBackwardOp>(*this);
    }
};

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
    bool need_grad_bias);

struct TileTorchEmbeddingDenseBackwardOp : TileGraph::OpNode
{
    TileGraph::TileNode *grad = nullptr;
    TileGraph::TileNode *indices = nullptr;
    TileGraph::TileNode *grad_weight = nullptr;

    TileTorchEmbeddingDenseBackwardOp() = default;
    TileTorchEmbeddingDenseBackwardOp(
        TileGraph::TileNode *grad_,
        TileGraph::TileNode *indices_,
        TileGraph::TileNode *grad_weight_) :
        grad(grad_), indices(indices_), grad_weight(grad_weight_)
    {
        inputs_ = {grad, indices};
        outputs_ = {grad_weight};
    }

    std::string op_name() const override
    {
        return "TILE_TORCH_EMBEDDING_DENSE_BACKWARD";
    }

    void execute(Runtime &runtime) const override;

    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileTorchEmbeddingDenseBackwardOp>(
            *this);
    }
};

void torch_embedding_dense_backward(
    TileGraph::TileNode *grad,
    TileGraph::TileNode *indices,
    TileGraph::TileNode *grad_weight);

struct TileTorchSdpaBackwardOp : TileGraph::OpNode
{
    bool is_causal = false;
    starpu::TorchDispatchArgs extra{};
    TileGraph::TileNode *q = nullptr;
    TileGraph::TileNode *k = nullptr;
    TileGraph::TileNode *v = nullptr;
    TileGraph::TileNode *grad_out = nullptr;
    TileGraph::TileNode *mask = nullptr;
    TileGraph::TileNode *grad_q = nullptr;
    TileGraph::TileNode *grad_k = nullptr;
    TileGraph::TileNode *grad_v = nullptr;

    TileTorchSdpaBackwardOp() = default;
    TileTorchSdpaBackwardOp(
        TileGraph::TileNode *q_,
        TileGraph::TileNode *k_,
        TileGraph::TileNode *v_,
        TileGraph::TileNode *grad_out_,
        TileGraph::TileNode *mask_,
        TileGraph::TileNode *grad_q_,
        TileGraph::TileNode *grad_k_,
        TileGraph::TileNode *grad_v_,
        bool is_causal_,
        starpu::TorchDispatchArgs extra_ = {}) :
        is_causal(is_causal_),
        extra(extra_),
        q(q_),
        k(k_),
        v(v_),
        grad_out(grad_out_),
        mask(mask_),
        grad_q(grad_q_),
        grad_k(grad_k_),
        grad_v(grad_v_)
    {
        inputs_ = {q, k, v, grad_out};
        if (mask != nullptr)
        {
            inputs_.push_back(mask);
        }
        outputs_ = {grad_q, grad_k, grad_v};
    }

    std::string op_name() const override
    {
        return "TILE_TORCH_SDPA_BACKWARD";
    }

    void execute(Runtime &runtime) const override;

    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileTorchSdpaBackwardOp>(*this);
    }
};

void torch_sdpa_backward(
    TileGraph::TileNode *q,
    TileGraph::TileNode *k,
    TileGraph::TileNode *v,
    TileGraph::TileNode *grad_out,
    TileGraph::TileNode *mask,
    TileGraph::TileNode *grad_q,
    TileGraph::TileNode *grad_k,
    TileGraph::TileNode *grad_v,
    bool is_causal = false,
    starpu::TorchDispatchArgs extra = {});

struct TileTorchNllLossForwardOp : TileGraph::OpNode
{
    Index reduction = 1;
    Index ignore_index = -100;
    TileGraph::TileNode *log_probs = nullptr;
    TileGraph::TileNode *target = nullptr;
    TileGraph::TileNode *loss = nullptr;
    TileGraph::TileNode *total_weight = nullptr;

    TileTorchNllLossForwardOp() = default;
    TileTorchNllLossForwardOp(
        TileGraph::TileNode *log_probs_,
        TileGraph::TileNode *target_,
        TileGraph::TileNode *loss_,
        TileGraph::TileNode *total_weight_,
        Index reduction_,
        Index ignore_index_) :
        reduction(reduction_),
        ignore_index(ignore_index_),
        log_probs(log_probs_),
        target(target_),
        loss(loss_),
        total_weight(total_weight_)
    {
        inputs_ = {log_probs, target};
        outputs_ = {loss, total_weight};
    }

    std::string op_name() const override
    {
        return "TILE_TORCH_NLL_LOSS_FORWARD";
    }

    void execute(Runtime &runtime) const override;

    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileTorchNllLossForwardOp>(*this);
    }
};

void torch_nll_loss_forward(
    TileGraph::TileNode *log_probs,
    TileGraph::TileNode *target,
    TileGraph::TileNode *loss,
    TileGraph::TileNode *total_weight,
    Index reduction,
    Index ignore_index);

struct TileTorchNllLossBackwardOp : TileGraph::OpNode
{
    Index reduction = 1;
    Index ignore_index = -100;
    TileGraph::TileNode *grad_output = nullptr;
    TileGraph::TileNode *log_probs = nullptr;
    TileGraph::TileNode *target = nullptr;
    TileGraph::TileNode *total_weight = nullptr;
    TileGraph::TileNode *grad_input = nullptr;

    TileTorchNllLossBackwardOp() = default;
    TileTorchNllLossBackwardOp(
        TileGraph::TileNode *grad_output_,
        TileGraph::TileNode *log_probs_,
        TileGraph::TileNode *target_,
        TileGraph::TileNode *total_weight_,
        TileGraph::TileNode *grad_input_,
        Index reduction_,
        Index ignore_index_) :
        reduction(reduction_),
        ignore_index(ignore_index_),
        grad_output(grad_output_),
        log_probs(log_probs_),
        target(target_),
        total_weight(total_weight_),
        grad_input(grad_input_)
    {
        inputs_ = {
            grad_output,
            log_probs,
            target,
            total_weight};
        outputs_ = {grad_input};
    }

    std::string op_name() const override
    {
        return "TILE_TORCH_NLL_LOSS_BACKWARD";
    }

    void execute(Runtime &runtime) const override;

    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileTorchNllLossBackwardOp>(*this);
    }
};

void torch_nll_loss_backward(
    TileGraph::TileNode *grad_output,
    TileGraph::TileNode *log_probs,
    TileGraph::TileNode *target,
    TileGraph::TileNode *total_weight,
    TileGraph::TileNode *grad_input,
    Index reduction,
    Index ignore_index);

} // namespace nntile::tile
