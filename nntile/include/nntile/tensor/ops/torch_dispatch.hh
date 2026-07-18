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
    starpu::TorchDispatchArgs extra{};
    TensorGraph::TensorNode *weight = nullptr;
    TensorGraph::TensorNode *indices = nullptr;
    TensorGraph::TensorNode *out = nullptr;

    TensorTorchEmbeddingOp() = default;
    TensorTorchEmbeddingOp(
        TensorGraph::TensorNode *weight_,
        TensorGraph::TensorNode *indices_,
        TensorGraph::TensorNode *out_,
        starpu::TorchDispatchArgs extra_ = {}) :
        extra(extra_),
        weight(weight_),
        indices(indices_),
        out(out_)
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
    const std::vector<Index> &out_shape,
    starpu::TorchDispatchArgs extra = {});

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

struct TensorTorchLayerNormBackwardOp : TensorGraph::OpNode
{
    Index normalized_ndim = 1;
    bool need_grad_input = false;
    bool need_grad_weight = false;
    bool need_grad_bias = false;
    TensorGraph::TensorNode *grad_out = nullptr;
    TensorGraph::TensorNode *input = nullptr;
    TensorGraph::TensorNode *mean = nullptr;
    TensorGraph::TensorNode *rstd = nullptr;
    TensorGraph::TensorNode *weight = nullptr;
    TensorGraph::TensorNode *bias = nullptr;
    TensorGraph::TensorNode *grad_input = nullptr;
    TensorGraph::TensorNode *grad_weight = nullptr;
    TensorGraph::TensorNode *grad_bias = nullptr;

    TensorTorchLayerNormBackwardOp() = default;
    TensorTorchLayerNormBackwardOp(
        TensorGraph::TensorNode *grad_out_,
        TensorGraph::TensorNode *input_,
        TensorGraph::TensorNode *mean_,
        TensorGraph::TensorNode *rstd_,
        TensorGraph::TensorNode *weight_,
        TensorGraph::TensorNode *bias_,
        TensorGraph::TensorNode *grad_input_,
        TensorGraph::TensorNode *grad_weight_,
        TensorGraph::TensorNode *grad_bias_,
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
        return "TORCH_NATIVE_LAYER_NORM_BACKWARD";
    }

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorTorchLayerNormBackwardOp>(
            *this);
    }

    void lower_to_tile(const LoweringContext &ctx) const override;
};

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
    bool need_grad_bias);

struct TensorTorchEmbeddingDenseBackwardOp : TensorGraph::OpNode
{
    starpu::TorchDispatchArgs extra{};
    TensorGraph::TensorNode *grad = nullptr;
    TensorGraph::TensorNode *indices = nullptr;
    TensorGraph::TensorNode *grad_weight = nullptr;

    TensorTorchEmbeddingDenseBackwardOp() = default;
    TensorTorchEmbeddingDenseBackwardOp(
        TensorGraph::TensorNode *grad_,
        TensorGraph::TensorNode *indices_,
        TensorGraph::TensorNode *grad_weight_,
        starpu::TorchDispatchArgs extra_ = {}) :
        extra(extra_),
        grad(grad_),
        indices(indices_),
        grad_weight(grad_weight_)
    {
        inputs_ = {grad, indices};
        outputs_ = {grad_weight};
    }

    std::string op_name() const override
    {
        return "TORCH_EMBEDDING_DENSE_BACKWARD";
    }

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorTorchEmbeddingDenseBackwardOp>(
            *this);
    }

    void lower_to_tile(const LoweringContext &ctx) const override;
};

void torch_embedding_dense_backward(
    TensorGraph::TensorNode *grad,
    TensorGraph::TensorNode *indices,
    TensorGraph::TensorNode *grad_weight,
    starpu::TorchDispatchArgs extra = {});

struct TensorTorchConvolutionOp : TensorGraph::OpNode
{
    starpu::TorchDispatchArgs extra{};
    TensorGraph::TensorNode *input = nullptr;
    TensorGraph::TensorNode *weight = nullptr;
    TensorGraph::TensorNode *bias = nullptr;
    TensorGraph::TensorNode *out = nullptr;

    TensorTorchConvolutionOp() = default;
    TensorTorchConvolutionOp(
        TensorGraph::TensorNode *input_,
        TensorGraph::TensorNode *weight_,
        TensorGraph::TensorNode *bias_,
        TensorGraph::TensorNode *out_,
        starpu::TorchDispatchArgs extra_ = {}) :
        extra(extra_),
        input(input_),
        weight(weight_),
        bias(bias_),
        out(out_)
    {
        inputs_ = {input, weight};
        if (bias != nullptr)
        {
            inputs_.push_back(bias);
        }
        outputs_ = {out};
    }

    std::string op_name() const override
    {
        return "TORCH_CONVOLUTION";
    }

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorTorchConvolutionOp>(*this);
    }

    void lower_to_tile(const LoweringContext &ctx) const override;
};

void torch_convolution(
    TensorGraph::TensorNode *input,
    TensorGraph::TensorNode *weight,
    TensorGraph::TensorNode *bias,
    TensorGraph::TensorNode *out,
    starpu::TorchDispatchArgs extra = {});

struct TensorTorchConvolutionBackwardOp : TensorGraph::OpNode
{
    starpu::TorchDispatchArgs extra{};
    bool need_grad_input = false;
    bool need_grad_weight = false;
    bool need_grad_bias = false;
    TensorGraph::TensorNode *grad_out = nullptr;
    TensorGraph::TensorNode *input = nullptr;
    TensorGraph::TensorNode *weight = nullptr;
    TensorGraph::TensorNode *grad_input = nullptr;
    TensorGraph::TensorNode *grad_weight = nullptr;
    TensorGraph::TensorNode *grad_bias = nullptr;

    TensorTorchConvolutionBackwardOp() = default;
    TensorTorchConvolutionBackwardOp(
        TensorGraph::TensorNode *grad_out_,
        TensorGraph::TensorNode *input_,
        TensorGraph::TensorNode *weight_,
        TensorGraph::TensorNode *grad_input_,
        TensorGraph::TensorNode *grad_weight_,
        TensorGraph::TensorNode *grad_bias_,
        bool need_grad_input_,
        bool need_grad_weight_,
        bool need_grad_bias_,
        starpu::TorchDispatchArgs extra_ = {}) :
        extra(extra_),
        need_grad_input(need_grad_input_),
        need_grad_weight(need_grad_weight_),
        need_grad_bias(need_grad_bias_),
        grad_out(grad_out_),
        input(input_),
        weight(weight_),
        grad_input(grad_input_),
        grad_weight(grad_weight_),
        grad_bias(grad_bias_)
    {
        inputs_ = {grad_out, input, weight};
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
        return "TORCH_CONVOLUTION_BACKWARD";
    }

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorTorchConvolutionBackwardOp>(*this);
    }

    void lower_to_tile(const LoweringContext &ctx) const override;
};

void torch_convolution_backward(
    TensorGraph::TensorNode *grad_out,
    TensorGraph::TensorNode *input,
    TensorGraph::TensorNode *weight,
    TensorGraph::TensorNode *grad_input,
    TensorGraph::TensorNode *grad_weight,
    TensorGraph::TensorNode *grad_bias,
    bool need_grad_input,
    bool need_grad_weight,
    bool need_grad_bias,
    starpu::TorchDispatchArgs extra = {});

struct TensorTorchMaxPool2dWithIndicesOp : TensorGraph::OpNode
{
    starpu::TorchDispatchArgs extra{};
    TensorGraph::TensorNode *input = nullptr;
    TensorGraph::TensorNode *out = nullptr;
    TensorGraph::TensorNode *indices = nullptr;

    TensorTorchMaxPool2dWithIndicesOp() = default;
    TensorTorchMaxPool2dWithIndicesOp(
        TensorGraph::TensorNode *input_,
        TensorGraph::TensorNode *out_,
        TensorGraph::TensorNode *indices_,
        starpu::TorchDispatchArgs extra_ = {}) :
        extra(extra_),
        input(input_),
        out(out_),
        indices(indices_)
    {
        inputs_ = {input};
        outputs_ = {out, indices};
    }

    std::string op_name() const override
    {
        return "TORCH_MAX_POOL2D_WITH_INDICES";
    }

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorTorchMaxPool2dWithIndicesOp>(*this);
    }

    void lower_to_tile(const LoweringContext &ctx) const override;
};

void torch_max_pool2d_with_indices(
    TensorGraph::TensorNode *input,
    TensorGraph::TensorNode *out,
    TensorGraph::TensorNode *indices,
    starpu::TorchDispatchArgs extra = {});

struct TensorTorchMaxPool2dWithIndicesBackwardOp : TensorGraph::OpNode
{
    starpu::TorchDispatchArgs extra{};
    TensorGraph::TensorNode *grad_out = nullptr;
    TensorGraph::TensorNode *input = nullptr;
    TensorGraph::TensorNode *indices = nullptr;
    TensorGraph::TensorNode *grad_input = nullptr;

    TensorTorchMaxPool2dWithIndicesBackwardOp() = default;
    TensorTorchMaxPool2dWithIndicesBackwardOp(
        TensorGraph::TensorNode *grad_out_,
        TensorGraph::TensorNode *input_,
        TensorGraph::TensorNode *indices_,
        TensorGraph::TensorNode *grad_input_,
        starpu::TorchDispatchArgs extra_ = {}) :
        extra(extra_),
        grad_out(grad_out_),
        input(input_),
        indices(indices_),
        grad_input(grad_input_)
    {
        inputs_ = {grad_out, input, indices};
        outputs_ = {grad_input};
    }

    std::string op_name() const override
    {
        return "TORCH_MAX_POOL2D_WITH_INDICES_BACKWARD";
    }

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<
            TensorTorchMaxPool2dWithIndicesBackwardOp>(*this);
    }

    void lower_to_tile(const LoweringContext &ctx) const override;
};

void torch_max_pool2d_with_indices_backward(
    TensorGraph::TensorNode *grad_out,
    TensorGraph::TensorNode *input,
    TensorGraph::TensorNode *indices,
    TensorGraph::TensorNode *grad_input,
    starpu::TorchDispatchArgs extra = {});

struct TensorTorchNativeBatchNormOp : TensorGraph::OpNode
{
    starpu::TorchDispatchArgs extra{};
    bool training = false;
    TensorGraph::TensorNode *input = nullptr;
    TensorGraph::TensorNode *weight = nullptr;
    TensorGraph::TensorNode *bias = nullptr;
    TensorGraph::TensorNode *running_mean = nullptr;
    TensorGraph::TensorNode *running_var = nullptr;
    TensorGraph::TensorNode *out = nullptr;
    TensorGraph::TensorNode *save_mean = nullptr;
    TensorGraph::TensorNode *save_invstd = nullptr;

    TensorTorchNativeBatchNormOp() = default;
    TensorTorchNativeBatchNormOp(
        TensorGraph::TensorNode *input_,
        TensorGraph::TensorNode *weight_,
        TensorGraph::TensorNode *bias_,
        TensorGraph::TensorNode *running_mean_,
        TensorGraph::TensorNode *running_var_,
        TensorGraph::TensorNode *out_,
        TensorGraph::TensorNode *save_mean_,
        TensorGraph::TensorNode *save_invstd_,
        bool training_,
        starpu::TorchDispatchArgs extra_ = {}) :
        extra(extra_),
        training(training_),
        input(input_),
        weight(weight_),
        bias(bias_),
        running_mean(running_mean_),
        running_var(running_var_),
        out(out_),
        save_mean(save_mean_),
        save_invstd(save_invstd_)
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
        if (running_mean != nullptr)
        {
            inputs_.push_back(running_mean);
        }
        if (running_var != nullptr)
        {
            inputs_.push_back(running_var);
        }
        outputs_ = {out, save_mean, save_invstd};
        if (training && running_mean != nullptr)
        {
            outputs_.push_back(running_mean);
        }
        if (training && running_var != nullptr)
        {
            outputs_.push_back(running_var);
        }
    }

    std::string op_name() const override
    {
        return "TORCH_NATIVE_BATCH_NORM";
    }

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorTorchNativeBatchNormOp>(*this);
    }

    void lower_to_tile(const LoweringContext &ctx) const override;
};

void torch_native_batch_norm(
    TensorGraph::TensorNode *input,
    TensorGraph::TensorNode *weight,
    TensorGraph::TensorNode *bias,
    TensorGraph::TensorNode *running_mean,
    TensorGraph::TensorNode *running_var,
    TensorGraph::TensorNode *out,
    TensorGraph::TensorNode *save_mean,
    TensorGraph::TensorNode *save_invstd,
    bool training,
    starpu::TorchDispatchArgs extra = {});

struct TensorTorchNativeBatchNormBackwardOp : TensorGraph::OpNode
{
    starpu::TorchDispatchArgs extra{};
    bool need_grad_input = false;
    bool need_grad_weight = false;
    bool need_grad_bias = false;
    TensorGraph::TensorNode *grad_out = nullptr;
    TensorGraph::TensorNode *input = nullptr;
    TensorGraph::TensorNode *weight = nullptr;
    TensorGraph::TensorNode *running_mean = nullptr;
    TensorGraph::TensorNode *running_var = nullptr;
    TensorGraph::TensorNode *save_mean = nullptr;
    TensorGraph::TensorNode *save_invstd = nullptr;
    TensorGraph::TensorNode *grad_input = nullptr;
    TensorGraph::TensorNode *grad_weight = nullptr;
    TensorGraph::TensorNode *grad_bias = nullptr;

    TensorTorchNativeBatchNormBackwardOp() = default;
    TensorTorchNativeBatchNormBackwardOp(
        TensorGraph::TensorNode *grad_out_,
        TensorGraph::TensorNode *input_,
        TensorGraph::TensorNode *weight_,
        TensorGraph::TensorNode *running_mean_,
        TensorGraph::TensorNode *running_var_,
        TensorGraph::TensorNode *save_mean_,
        TensorGraph::TensorNode *save_invstd_,
        TensorGraph::TensorNode *grad_input_,
        TensorGraph::TensorNode *grad_weight_,
        TensorGraph::TensorNode *grad_bias_,
        bool need_grad_input_,
        bool need_grad_weight_,
        bool need_grad_bias_,
        starpu::TorchDispatchArgs extra_ = {}) :
        extra(extra_),
        need_grad_input(need_grad_input_),
        need_grad_weight(need_grad_weight_),
        need_grad_bias(need_grad_bias_),
        grad_out(grad_out_),
        input(input_),
        weight(weight_),
        running_mean(running_mean_),
        running_var(running_var_),
        save_mean(save_mean_),
        save_invstd(save_invstd_),
        grad_input(grad_input_),
        grad_weight(grad_weight_),
        grad_bias(grad_bias_)
    {
        inputs_ = {grad_out, input};
        if (weight != nullptr)
        {
            inputs_.push_back(weight);
        }
        if (running_mean != nullptr)
        {
            inputs_.push_back(running_mean);
        }
        if (running_var != nullptr)
        {
            inputs_.push_back(running_var);
        }
        if (save_mean != nullptr)
        {
            inputs_.push_back(save_mean);
        }
        if (save_invstd != nullptr)
        {
            inputs_.push_back(save_invstd);
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
        return "TORCH_NATIVE_BATCH_NORM_BACKWARD";
    }

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorTorchNativeBatchNormBackwardOp>(*this);
    }

    void lower_to_tile(const LoweringContext &ctx) const override;
};

void torch_native_batch_norm_backward(
    TensorGraph::TensorNode *grad_out,
    TensorGraph::TensorNode *input,
    TensorGraph::TensorNode *weight,
    TensorGraph::TensorNode *running_mean,
    TensorGraph::TensorNode *running_var,
    TensorGraph::TensorNode *save_mean,
    TensorGraph::TensorNode *save_invstd,
    TensorGraph::TensorNode *grad_input,
    TensorGraph::TensorNode *grad_weight,
    TensorGraph::TensorNode *grad_bias,
    bool need_grad_input,
    bool need_grad_weight,
    bool need_grad_bias,
    starpu::TorchDispatchArgs extra = {});

struct TensorTorchSdpaBackwardOp : TensorGraph::OpNode
{
    bool is_causal = false;
    starpu::TorchDispatchArgs extra{};
    TensorGraph::TensorNode *q = nullptr;
    TensorGraph::TensorNode *k = nullptr;
    TensorGraph::TensorNode *v = nullptr;
    TensorGraph::TensorNode *grad_out = nullptr;
    TensorGraph::TensorNode *mask = nullptr;
    TensorGraph::TensorNode *grad_q = nullptr;
    TensorGraph::TensorNode *grad_k = nullptr;
    TensorGraph::TensorNode *grad_v = nullptr;

    TensorTorchSdpaBackwardOp() = default;
    TensorTorchSdpaBackwardOp(
        TensorGraph::TensorNode *q_,
        TensorGraph::TensorNode *k_,
        TensorGraph::TensorNode *v_,
        TensorGraph::TensorNode *grad_out_,
        TensorGraph::TensorNode *mask_,
        TensorGraph::TensorNode *grad_q_,
        TensorGraph::TensorNode *grad_k_,
        TensorGraph::TensorNode *grad_v_,
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
        return "TORCH_SDPA_BACKWARD";
    }

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorTorchSdpaBackwardOp>(*this);
    }

    void lower_to_tile(const LoweringContext &ctx) const override;
};

void torch_sdpa_backward(
    TensorGraph::TensorNode *q,
    TensorGraph::TensorNode *k,
    TensorGraph::TensorNode *v,
    TensorGraph::TensorNode *grad_out,
    TensorGraph::TensorNode *mask,
    TensorGraph::TensorNode *grad_q,
    TensorGraph::TensorNode *grad_k,
    TensorGraph::TensorNode *grad_v,
    bool is_causal = false,
    starpu::TorchDispatchArgs extra = {});

struct TensorTorchNllLossForwardOp : TensorGraph::OpNode
{
    Index reduction = 1;
    Index ignore_index = -100;
    starpu::TorchDispatchArgs extra{};
    TensorGraph::TensorNode *log_probs = nullptr;
    TensorGraph::TensorNode *target = nullptr;
    TensorGraph::TensorNode *loss = nullptr;
    TensorGraph::TensorNode *total_weight = nullptr;

    TensorTorchNllLossForwardOp() = default;
    TensorTorchNllLossForwardOp(
        TensorGraph::TensorNode *log_probs_,
        TensorGraph::TensorNode *target_,
        TensorGraph::TensorNode *loss_,
        TensorGraph::TensorNode *total_weight_,
        Index reduction_,
        Index ignore_index_,
        starpu::TorchDispatchArgs extra_ = {}) :
        reduction(reduction_),
        ignore_index(ignore_index_),
        extra(extra_),
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
        return "TORCH_NLL_LOSS_FORWARD";
    }

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorTorchNllLossForwardOp>(*this);
    }

    void lower_to_tile(const LoweringContext &ctx) const override;
};

void torch_nll_loss_forward(
    TensorGraph::TensorNode *log_probs,
    TensorGraph::TensorNode *target,
    TensorGraph::TensorNode *loss,
    TensorGraph::TensorNode *total_weight,
    Index reduction,
    Index ignore_index,
    starpu::TorchDispatchArgs extra = {});

struct TensorTorchNllLossBackwardOp : TensorGraph::OpNode
{
    Index reduction = 1;
    Index ignore_index = -100;
    starpu::TorchDispatchArgs extra{};
    TensorGraph::TensorNode *grad_output = nullptr;
    TensorGraph::TensorNode *log_probs = nullptr;
    TensorGraph::TensorNode *target = nullptr;
    TensorGraph::TensorNode *total_weight = nullptr;
    TensorGraph::TensorNode *grad_input = nullptr;

    TensorTorchNllLossBackwardOp() = default;
    TensorTorchNllLossBackwardOp(
        TensorGraph::TensorNode *grad_output_,
        TensorGraph::TensorNode *log_probs_,
        TensorGraph::TensorNode *target_,
        TensorGraph::TensorNode *total_weight_,
        TensorGraph::TensorNode *grad_input_,
        Index reduction_,
        Index ignore_index_,
        starpu::TorchDispatchArgs extra_ = {}) :
        reduction(reduction_),
        ignore_index(ignore_index_),
        extra(extra_),
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
        return "TORCH_NLL_LOSS_BACKWARD";
    }

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorTorchNllLossBackwardOp>(*this);
    }

    void lower_to_tile(const LoweringContext &ctx) const override;
};

void torch_nll_loss_backward(
    TensorGraph::TensorNode *grad_output,
    TensorGraph::TensorNode *log_probs,
    TensorGraph::TensorNode *target,
    TensorGraph::TensorNode *total_weight,
    TensorGraph::TensorNode *grad_input,
    Index reduction,
    Index ignore_index,
    starpu::TorchDispatchArgs extra = {});

} // namespace nntile::tensor
