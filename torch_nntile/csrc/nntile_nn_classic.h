/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_nn_classic.h
 * Python-facing classic NNTile nn ops (torch_nntile.nn path).
 */

#pragma once

#include <ATen/Functions.h>
#include <ATen/Tensor.h>
#include <ATen/TensorUtils.h>
#include <torch/csrc/autograd/custom_function.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <optional>
#include <tuple>
#include <vector>

namespace torch_nntile
{
namespace nn_classic
{

at::Tensor add_forward(
    const at::Tensor &x,
    const at::Tensor &y,
    double alpha = 1.0,
    double beta = 1.0);

std::tuple<at::Tensor, at::Tensor> add_backward(
    const at::Tensor &grad_out,
    std::array<bool, 2> output_mask,
    double alpha = 1.0,
    double beta = 1.0);

at::Tensor mul_forward(const at::Tensor &a, const at::Tensor &b);

std::tuple<at::Tensor, at::Tensor> mul_backward(
    const at::Tensor &grad_out,
    const at::Tensor &a,
    const at::Tensor &b,
    std::array<bool, 2> output_mask);

at::Tensor relu_forward(const at::Tensor &input);

//! ``saved`` is the ReLU output (mask), matching ``relu_backward``.
at::Tensor relu_backward(
    const at::Tensor &saved_output,
    const at::Tensor &grad_out);

at::Tensor silu_forward(const at::Tensor &input);

//! ``input`` is the pre-activation (not the SiLU output).
at::Tensor silu_backward(
    const at::Tensor &input,
    const at::Tensor &grad_out);

at::Tensor gelu_forward(
    const at::Tensor &input,
    bool approximate_tanh = true);

at::Tensor gelu_backward(
    const at::Tensor &input,
    const at::Tensor &grad_out,
    bool approximate_tanh = true);

at::Tensor mul_scalar_forward(
    const at::Tensor &input,
    double scalar);

at::Tensor mul_scalar_backward(
    const at::Tensor &grad_out,
    double scalar);

std::tuple<at::Tensor, at::Tensor, at::Tensor> layer_norm_forward(
    const at::Tensor &input,
    at::IntArrayRef normalized_shape,
    const std::optional<at::Tensor> &weight,
    const std::optional<at::Tensor> &bias,
    double eps);

std::tuple<at::Tensor, at::Tensor, at::Tensor> layer_norm_backward(
    const at::Tensor &grad_out,
    const at::Tensor &input,
    at::IntArrayRef normalized_shape,
    const at::Tensor &mean,
    const at::Tensor &rstd,
    const std::optional<at::Tensor> &weight,
    const std::optional<at::Tensor> &bias,
    std::array<bool, 3> output_mask);

at::Tensor embedding_forward(
    const at::Tensor &weight,
    const at::Tensor &indices);

at::Tensor embedding_backward(
    const at::Tensor &grad_output,
    const at::Tensor &indices,
    int64_t num_weights);

at::Tensor scale_slice_forward(
    const at::Tensor &input,
    int64_t axis,
    int64_t axis_size,
    double alpha);

at::Tensor scale_slice_backward(
    const at::Tensor &grad_out,
    int64_t axis,
    double alpha);

at::Tensor cat_forward(
    const at::Tensor &a,
    const at::Tensor &b,
    int64_t dim);

std::tuple<at::Tensor, at::Tensor> cat_backward(
    const at::Tensor &grad_out,
    int64_t dim,
    int64_t a_size,
    int64_t b_size,
    std::array<bool, 2> output_mask);

at::Tensor narrow_forward(
    const at::Tensor &input,
    int64_t dim,
    int64_t start,
    int64_t length);

at::Tensor narrow_backward(
    const at::Tensor &grad_out,
    at::IntArrayRef input_sizes,
    int64_t dim,
    int64_t start);

namespace detail
{

inline at::Tensor save_int64_list(std::vector<int64_t> const &values)
{
    auto opts = at::TensorOptions()
        .dtype(at::kLong)
        .device(at::kCPU);
    at::Tensor t = at::empty(
        {static_cast<int64_t>(values.size())},
        opts);
    if (!values.empty())
    {
        std::memcpy(
            t.data_ptr<int64_t>(),
            values.data(),
            values.size() * sizeof(int64_t));
    }
    return t;
}

inline std::vector<int64_t> load_int64_list(c10::IValue const &v)
{
    at::Tensor t = v.toTensor().contiguous();
    int64_t const n = t.numel();
    int64_t const *ptr = t.data_ptr<int64_t>();
    return std::vector<int64_t>(ptr, ptr + n);
}

class AddFn : public torch::autograd::Function<AddFn>
{
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        at::Tensor x,
        at::Tensor y,
        double alpha,
        double beta)
    {
        ctx->saved_data["alpha"] = alpha;
        ctx->saved_data["beta"] = beta;
        // d(alpha x + beta y) = alpha dZ, beta dZ; payloads unused.
        return nn_classic::add_forward(x, y, alpha, beta);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs)
    {
        double const alpha = ctx->saved_data["alpha"].toDouble();
        double const beta = ctx->saved_data["beta"].toDouble();
        std::array<bool, 2> const mask = {
            ctx->needs_input_grad(0),
            ctx->needs_input_grad(1),
        };
        auto gb = nn_classic::add_backward(
            grad_outputs[0],
            mask,
            alpha,
            beta);
        return {std::get<0>(gb), std::get<1>(gb), {}, {}};
    }
};

class MulFn : public torch::autograd::Function<MulFn>
{
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        at::Tensor a,
        at::Tensor b)
    {
        ctx->save_for_backward({a, b});
        return nn_classic::mul_forward(a, b);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs)
    {
        auto saved = ctx->get_saved_variables();
        std::array<bool, 2> const mask = {
            ctx->needs_input_grad(0),
            ctx->needs_input_grad(1),
        };
        auto gb = nn_classic::mul_backward(
            grad_outputs[0],
            saved[0],
            saved[1],
            mask);
        return {std::get<0>(gb), std::get<1>(gb)};
    }
};

class MulScalarFn : public torch::autograd::Function<MulScalarFn>
{
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        at::Tensor input,
        double scalar)
    {
        ctx->saved_data["scalar"] = scalar;
        return nn_classic::mul_scalar_forward(input, scalar);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs)
    {
        double const scalar = ctx->saved_data["scalar"].toDouble();
        at::Tensor grad;
        if (ctx->needs_input_grad(0))
        {
            grad = nn_classic::mul_scalar_backward(
                grad_outputs[0],
                scalar);
        }
        return {grad, {}};
    }
};

class ReluFn : public torch::autograd::Function<ReluFn>
{
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        at::Tensor input)
    {
        at::Tensor out = nn_classic::relu_forward(input);
        // ReluBackward0 saves the result; mask is (y > 0), not the input.
        ctx->save_for_backward({out});
        return out;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs)
    {
        auto saved = ctx->get_saved_variables();
        at::Tensor grad;
        if (ctx->needs_input_grad(0))
        {
            grad = nn_classic::relu_backward(
                saved[0],
                grad_outputs[0]);
        }
        return {grad};
    }
};

class SiluFn : public torch::autograd::Function<SiluFn>
{
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        at::Tensor input)
    {
        ctx->save_for_backward({input});
        return nn_classic::silu_forward(input);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs)
    {
        auto saved = ctx->get_saved_variables();
        at::Tensor grad;
        if (ctx->needs_input_grad(0))
        {
            grad = nn_classic::silu_backward(
                saved[0],
                grad_outputs[0]);
        }
        return {grad};
    }
};

class GeluFn : public torch::autograd::Function<GeluFn>
{
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        at::Tensor input,
        bool approximate_tanh)
    {
        ctx->saved_data["approximate_tanh"] = approximate_tanh;
        ctx->save_for_backward({input});
        return nn_classic::gelu_forward(input, approximate_tanh);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs)
    {
        auto saved = ctx->get_saved_variables();
        bool const approximate_tanh =
            ctx->saved_data["approximate_tanh"].toBool();
        at::Tensor grad;
        if (ctx->needs_input_grad(0))
        {
            grad = nn_classic::gelu_backward(
                saved[0],
                grad_outputs[0],
                approximate_tanh);
        }
        return {grad, {}};
    }
};

class LayerNormFn : public torch::autograd::Function<LayerNormFn>
{
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        at::Tensor input,
        at::Tensor weight,
        at::Tensor bias,
        double eps)
    {
        std::vector<int64_t> normalized_shape;
        if (weight.defined())
        {
            normalized_shape.assign(
                weight.sizes().begin(),
                weight.sizes().end());
        }
        else
        {
            normalized_shape.push_back(input.size(-1));
        }
        ctx->saved_data["normalized_shape"] = save_int64_list(
            normalized_shape);
        ctx->saved_data["eps"] = eps;
        ctx->saved_data["has_weight"] = weight.defined();
        ctx->saved_data["has_bias"] = bias.defined();
        std::optional<at::Tensor> weight_opt;
        std::optional<at::Tensor> bias_opt;
        if (weight.defined())
        {
            weight_opt = weight;
        }
        if (bias.defined())
        {
            bias_opt = bias;
        }
        auto result = nn_classic::layer_norm_forward(
            input,
            normalized_shape,
            weight_opt,
            bias_opt,
            eps);
        ctx->save_for_backward({
            input,
            std::get<1>(result),
            std::get<2>(result),
            weight,
            bias,
        });
        return std::get<0>(result);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs)
    {
        auto saved = ctx->get_saved_variables();
        bool const has_weight = ctx->saved_data["has_weight"].toBool();
        bool const has_bias = ctx->saved_data["has_bias"].toBool();
        std::vector<int64_t> normalized_shape = load_int64_list(
            ctx->saved_data["normalized_shape"]);
        std::optional<at::Tensor> weight;
        std::optional<at::Tensor> bias;
        if (has_weight)
        {
            weight = saved[3];
        }
        if (has_bias)
        {
            bias = saved[4];
        }
        auto gb = nn_classic::layer_norm_backward(
            grad_outputs[0],
            saved[0],
            normalized_shape,
            saved[1],
            saved[2],
            weight,
            bias,
            {
                ctx->needs_input_grad(0),
                has_weight && ctx->needs_input_grad(1),
                has_bias && ctx->needs_input_grad(2),
            });
        return {
            std::get<0>(gb),
            std::get<1>(gb),
            std::get<2>(gb),
            {},
        };
    }
};

class EmbeddingFn : public torch::autograd::Function<EmbeddingFn>
{
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        at::Tensor weight,
        at::Tensor indices)
    {
        ctx->save_for_backward({indices});
        ctx->saved_data["num_weights"] = weight.size(0);
        return nn_classic::embedding_forward(weight, indices);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs)
    {
        auto saved = ctx->get_saved_variables();
        at::Tensor grad_weight;
        if (ctx->needs_input_grad(0))
        {
            grad_weight = nn_classic::embedding_backward(
                grad_outputs[0],
                saved[0],
                ctx->saved_data["num_weights"].toInt());
        }
        return {grad_weight, {}};
    }
};

class ScaleSliceFn : public torch::autograd::Function<ScaleSliceFn>
{
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        at::Tensor input,
        int64_t axis,
        int64_t axis_size,
        double alpha)
    {
        ctx->saved_data["axis"] = axis;
        ctx->saved_data["alpha"] = alpha;
        return nn_classic::scale_slice_forward(
            input,
            axis,
            axis_size,
            alpha);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs)
    {
        at::Tensor grad;
        if (ctx->needs_input_grad(0))
        {
            grad = nn_classic::scale_slice_backward(
                grad_outputs[0],
                ctx->saved_data["axis"].toInt(),
                ctx->saved_data["alpha"].toDouble());
        }
        return {grad, {}, {}, {}};
    }
};

class CatFn : public torch::autograd::Function<CatFn>
{
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        at::Tensor a,
        at::Tensor b,
        int64_t dim)
    {
        int64_t const wrapped = at::maybe_wrap_dim(dim, a.dim());
        ctx->saved_data["dim"] = wrapped;
        ctx->saved_data["a_size"] = a.size(wrapped);
        ctx->saved_data["b_size"] = b.size(wrapped);
        return nn_classic::cat_forward(a, b, wrapped);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs)
    {
        std::array<bool, 2> const mask = {
            ctx->needs_input_grad(0),
            ctx->needs_input_grad(1),
        };
        auto gb = nn_classic::cat_backward(
            grad_outputs[0],
            ctx->saved_data["dim"].toInt(),
            ctx->saved_data["a_size"].toInt(),
            ctx->saved_data["b_size"].toInt(),
            mask);
        return {std::get<0>(gb), std::get<1>(gb), {}};
    }
};

class NarrowFn : public torch::autograd::Function<NarrowFn>
{
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        at::Tensor input,
        int64_t dim,
        int64_t start,
        int64_t length)
    {
        int64_t const wrapped = at::maybe_wrap_dim(dim, input.dim());
        ctx->saved_data["dim"] = wrapped;
        ctx->saved_data["start"] = start;
        ctx->saved_data["sizes"] = save_int64_list(input.sizes().vec());
        return nn_classic::narrow_forward(
            input,
            wrapped,
            start,
            length);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs)
    {
        at::Tensor grad;
        if (ctx->needs_input_grad(0))
        {
            grad = nn_classic::narrow_backward(
                grad_outputs[0],
                load_int64_list(ctx->saved_data["sizes"]),
                ctx->saved_data["dim"].toInt(),
                ctx->saved_data["start"].toInt());
        }
        return {grad, {}, {}, {}};
    }
};

} // namespace detail

inline at::Tensor add(
    const at::Tensor &x,
    const at::Tensor &y,
    double alpha = 1.0,
    double beta = 1.0)
{
    return detail::AddFn::apply(x, y, alpha, beta);
}

inline at::Tensor mul(const at::Tensor &a, const at::Tensor &b)
{
    return detail::MulFn::apply(a, b);
}

inline at::Tensor mul_scalar(const at::Tensor &input, double scalar)
{
    return detail::MulScalarFn::apply(input, scalar);
}

inline at::Tensor relu(const at::Tensor &input)
{
    return detail::ReluFn::apply(input);
}

inline at::Tensor silu(const at::Tensor &input)
{
    return detail::SiluFn::apply(input);
}

inline at::Tensor gelu(
    const at::Tensor &input,
    bool approximate_tanh = true)
{
    return detail::GeluFn::apply(input, approximate_tanh);
}

inline at::Tensor layer_norm(
    const at::Tensor &input,
    at::IntArrayRef normalized_shape,
    const std::optional<at::Tensor> &weight = std::nullopt,
    const std::optional<at::Tensor> &bias = std::nullopt,
    double eps = 1e-5)
{
    return detail::LayerNormFn::apply(
        input,
        weight.value_or(at::Tensor()),
        bias.value_or(at::Tensor()),
        eps);
}

inline at::Tensor nn_embedding(
    const at::Tensor &weight,
    const at::Tensor &indices)
{
    return detail::EmbeddingFn::apply(weight, indices);
}

inline at::Tensor scale_slice(
    const at::Tensor &input,
    int64_t axis,
    int64_t axis_size,
    double alpha = 1.0)
{
    return detail::ScaleSliceFn::apply(
        input,
        axis,
        axis_size,
        alpha);
}

inline at::Tensor cat(
    const at::Tensor &a,
    const at::Tensor &b,
    int64_t dim)
{
    return detail::CatFn::apply(a, b, dim);
}

inline at::Tensor narrow(
    const at::Tensor &input,
    int64_t dim,
    int64_t start,
    int64_t length)
{
    return detail::NarrowFn::apply(input, dim, start, length);
}

} // namespace nn_classic
} // namespace torch_nntile
