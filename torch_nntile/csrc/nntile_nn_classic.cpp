/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_nn_classic.cpp
 */

#include "nntile_nn_classic.h"

#include "nntile_executor_classic.h"
#include "nntile_graph_recorder_impl.h"

#include <torch_nntile/classic_nn.hh>

#include <ATen/Functions.h>
#include <ATen/TensorUtils.h>

#include <cmath>
#include <vector>

namespace torch_nntile
{
namespace nn_classic
{

namespace
{

bool is_nntile_device(c10::Device device)
{
    return device.type() == c10::DeviceType::PrivateUse1;
}

void check_fp32_contiguous_nntile(
    const at::Tensor &tensor,
    const char *name)
{
    TORCH_CHECK(
        is_nntile_device(tensor.device()),
        "nntile ",
        name,
        ": expected nntile tensor");
    TORCH_CHECK(
        tensor.scalar_type() == at::ScalarType::Float,
        "nntile ",
        name,
        " supports float32 only");
    TORCH_CHECK(
        tensor.is_contiguous(),
        "nntile ",
        name,
        " requires contiguous");
}

int64_t resolve_norm_axis(
    c10::IntArrayRef input_shape,
    c10::IntArrayRef normalized_shape)
{
    TORCH_CHECK(
        normalized_shape.size() == 1,
        "nntile layer_norm supports a single normalized dimension");
    TORCH_CHECK(
        input_shape.size() >= normalized_shape.size(),
        "nntile layer_norm: input rank too small");
    const int64_t axis = static_cast<int64_t>(input_shape.size()) -
        static_cast<int64_t>(normalized_shape.size());
    for (std::size_t i = 0; i < normalized_shape.size(); ++i)
    {
        TORCH_CHECK(
            input_shape[static_cast<std::size_t>(axis) + i] ==
                normalized_shape[i],
            "nntile layer_norm: normalized_shape mismatch");
    }
    return axis;
}

std::vector<int64_t> reduced_sizes(
    c10::IntArrayRef input_shape,
    int64_t axis)
{
    std::vector<int64_t> sizes;
    sizes.reserve(static_cast<std::size_t>(input_shape.size()));
    for (int64_t i = 0; i < static_cast<int64_t>(input_shape.size()); ++i)
    {
        if (i != axis)
        {
            sizes.push_back(input_shape[static_cast<std::size_t>(i)]);
        }
    }
    return sizes;
}

std::optional<at::Tensor> optional_defined(
    const std::optional<at::Tensor> &tensor,
    const char *name)
{
    if (!tensor.has_value() || !tensor->defined())
    {
        return std::nullopt;
    }
    check_fp32_contiguous_nntile(*tensor, name);
    return *tensor;
}

std::vector<int64_t> embedding_output_shape(
    c10::IntArrayRef index_shape,
    int64_t embed_dim)
{
    std::vector<int64_t> out_shape(
        index_shape.begin(),
        index_shape.end());
    out_shape.push_back(embed_dim);
    return out_shape;
}

} // namespace

at::Tensor add_forward(
    const at::Tensor &x,
    const at::Tensor &y,
    double alpha,
    double beta)
{
    nntile::GraphFillScope record;
    check_fp32_contiguous_nntile(x, "add");
    check_fp32_contiguous_nntile(y, "add");
    TORCH_CHECK(x.sizes().equals(y.sizes()), "nntile add: shape mismatch");
    at::Tensor out = at::empty_like(x);
    classic_tensor_add_fp32(
        static_cast<float>(alpha),
        x,
        static_cast<float>(beta),
        y,
        out);
    return out;
}

std::tuple<at::Tensor, at::Tensor> add_backward(
    const at::Tensor &grad_out,
    std::array<bool, 2> output_mask,
    double alpha,
    double beta)
{
    nntile::GraphFillScope record;
    check_fp32_contiguous_nntile(grad_out, "add_backward");
    at::Tensor grad_x;
    at::Tensor grad_y;
    if (output_mask[0])
    {
        grad_x = at::empty_like(grad_out);
        classic_tensor_mul_scalar_fp32(
            grad_out,
            grad_x,
            static_cast<float>(alpha));
    }
    if (output_mask[1])
    {
        grad_y = at::empty_like(grad_out);
        classic_tensor_mul_scalar_fp32(
            grad_out,
            grad_y,
            static_cast<float>(beta));
    }
    return {grad_x, grad_y};
}

at::Tensor mul_forward(const at::Tensor &a, const at::Tensor &b)
{
    nntile::GraphFillScope record;
    check_fp32_contiguous_nntile(a, "mul");
    check_fp32_contiguous_nntile(b, "mul");
    TORCH_CHECK(a.sizes().equals(b.sizes()), "nntile mul: shape mismatch");
    at::Tensor out = at::empty_like(a);
    classic_tensor_mul_fp32(a, b, out);
    return out;
}

std::tuple<at::Tensor, at::Tensor> mul_backward(
    const at::Tensor &grad_out,
    const at::Tensor &a,
    const at::Tensor &b,
    std::array<bool, 2> output_mask)
{
    nntile::GraphFillScope record;
    check_fp32_contiguous_nntile(grad_out, "mul_backward");
    at::Tensor grad_a;
    at::Tensor grad_b;
    if (output_mask[0])
    {
        grad_a = at::empty_like(a);
        classic_tensor_mul_fp32(grad_out, b, grad_a);
    }
    if (output_mask[1])
    {
        grad_b = at::empty_like(b);
        classic_tensor_mul_fp32(grad_out, a, grad_b);
    }
    return {grad_a, grad_b};
}

at::Tensor mul_scalar_forward(
    const at::Tensor &input,
    double scalar)
{
    nntile::GraphFillScope record;
    check_fp32_contiguous_nntile(input, "mul_scalar");
    at::Tensor out = at::empty_like(input);
    classic_tensor_mul_scalar_fp32(
        input,
        out,
        static_cast<float>(scalar));
    return out;
}

at::Tensor mul_scalar_backward(
    const at::Tensor &grad_out,
    double scalar)
{
    nntile::GraphFillScope record;
    check_fp32_contiguous_nntile(grad_out, "mul_scalar_backward");
    at::Tensor grad_input = at::empty_like(grad_out);
    classic_tensor_mul_scalar_fp32(
        grad_out,
        grad_input,
        static_cast<float>(scalar));
    return grad_input;
}

at::Tensor relu_forward(const at::Tensor &input)
{
    nntile::GraphFillScope record;
    check_fp32_contiguous_nntile(input, "relu");
    at::Tensor out = at::empty_like(input);
    classic_tensor_relu_fp32(input, out);
    return out;
}

at::Tensor relu_backward(
    const at::Tensor &saved_output,
    const at::Tensor &grad_out)
{
    nntile::GraphFillScope record;
    check_fp32_contiguous_nntile(saved_output, "relu_backward");
    check_fp32_contiguous_nntile(grad_out, "relu_backward");
    at::Tensor grad_input = at::empty_like(saved_output);
    classic_tensor_relu_backward_fp32(saved_output, grad_out, grad_input);
    return grad_input;
}

at::Tensor silu_forward(const at::Tensor &input)
{
    nntile::GraphFillScope record;
    check_fp32_contiguous_nntile(input, "silu");
    at::Tensor out = at::empty_like(input);
    classic_tensor_silu_fp32(input, out);
    return out;
}

at::Tensor silu_backward(
    const at::Tensor &input,
    const at::Tensor &grad_out)
{
    nntile::GraphFillScope record;
    check_fp32_contiguous_nntile(input, "silu_backward");
    check_fp32_contiguous_nntile(grad_out, "silu_backward");
    at::Tensor grad_input = at::empty_like(input);
    classic_tensor_silu_backward_fp32(input, grad_out, grad_input);
    return grad_input;
}

at::Tensor gelu_forward(
    const at::Tensor &input,
    bool approximate_tanh)
{
    nntile::GraphFillScope record;
    check_fp32_contiguous_nntile(input, "gelu");
    at::Tensor out = at::empty_like(input);
    classic_tensor_gelu_fp32(input, out, approximate_tanh);
    return out;
}

at::Tensor gelu_backward(
    const at::Tensor &input,
    const at::Tensor &grad_out,
    bool approximate_tanh)
{
    nntile::GraphFillScope record;
    check_fp32_contiguous_nntile(input, "gelu_backward");
    check_fp32_contiguous_nntile(grad_out, "gelu_backward");
    at::Tensor grad_input = at::empty_like(input);
    classic_tensor_gelu_backward_fp32(
        input,
        grad_out,
        grad_input,
        approximate_tanh);
    return grad_input;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> layer_norm_forward(
    const at::Tensor &input,
    at::IntArrayRef normalized_shape,
    const std::optional<at::Tensor> &weight,
    const std::optional<at::Tensor> &bias,
    double eps)
{
    nntile::GraphFillScope record;
    check_fp32_contiguous_nntile(input, "layer_norm");
    const int64_t norm_axis = resolve_norm_axis(
        input.sizes(),
        normalized_shape);
    std::optional<at::Tensor> weight_t = optional_defined(weight, "weight");
    std::optional<at::Tensor> bias_t = optional_defined(bias, "bias");
    if (weight_t.has_value())
    {
        TORCH_CHECK(
            weight_t->dim() == 1 &&
                weight_t->size(0) == input.size(norm_axis),
            "nntile layer_norm: invalid weight shape");
    }
    if (bias_t.has_value())
    {
        TORCH_CHECK(
            bias_t->dim() == 1 && bias_t->size(0) == input.size(norm_axis),
            "nntile layer_norm: invalid bias shape");
    }

    at::Tensor output = at::empty_like(input);
    at::Tensor mean = at::empty(
        reduced_sizes(input.sizes(), norm_axis),
        input.options().memory_format(at::MemoryFormat::Contiguous));
    at::Tensor rstd = at::empty(
        reduced_sizes(input.sizes(), norm_axis),
        input.options().memory_format(at::MemoryFormat::Contiguous));
    classic_tensor_layer_norm_forward_fp32(
        input,
        weight_t.has_value() ? &*weight_t : nullptr,
        bias_t.has_value() ? &*bias_t : nullptr,
        weight_t.has_value(),
        bias_t.has_value(),
        output,
        mean,
        rstd,
        norm_axis,
        static_cast<float>(eps));
    return {output, mean, rstd};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> layer_norm_backward(
    const at::Tensor &grad_out,
    const at::Tensor &input,
    at::IntArrayRef normalized_shape,
    const at::Tensor &mean,
    const at::Tensor &rstd,
    const std::optional<at::Tensor> &weight,
    const std::optional<at::Tensor> &bias,
    std::array<bool, 3> output_mask)
{
    nntile::GraphFillScope record;
    check_fp32_contiguous_nntile(grad_out, "layer_norm_backward");
    check_fp32_contiguous_nntile(input, "layer_norm_backward");
    check_fp32_contiguous_nntile(mean, "layer_norm_backward");
    check_fp32_contiguous_nntile(rstd, "layer_norm_backward");
    const int64_t norm_axis = resolve_norm_axis(
        input.sizes(),
        normalized_shape);
    std::optional<at::Tensor> weight_t = optional_defined(weight, "weight");
    std::optional<at::Tensor> bias_t = optional_defined(bias, "bias");

    at::Tensor grad_input;
    at::Tensor grad_weight;
    at::Tensor grad_bias;
    if (output_mask[0])
    {
        grad_input = at::empty_like(input);
    }
    if (output_mask[1] && weight_t.has_value())
    {
        grad_weight = at::empty_like(*weight_t);
    }
    if (output_mask[2] && bias_t.has_value())
    {
        grad_bias = at::empty_like(*bias_t);
    }

    classic_tensor_layer_norm_backward_fp32(
        grad_out,
        input,
        mean,
        rstd,
        weight_t.has_value() ? &*weight_t : nullptr,
        bias_t.has_value() ? &*bias_t : nullptr,
        weight_t.has_value(),
        bias_t.has_value(),
        output_mask[0] ? &grad_input : nullptr,
        output_mask[1] && weight_t.has_value() ? &grad_weight : nullptr,
        output_mask[2] && bias_t.has_value() ? &grad_bias : nullptr,
        output_mask[0],
        output_mask[1] && weight_t.has_value(),
        output_mask[2] && bias_t.has_value(),
        norm_axis);
    return {grad_input, grad_weight, grad_bias};
}

at::Tensor embedding_forward(
    const at::Tensor &weight,
    const at::Tensor &indices)
{
    nntile::GraphFillScope record;
    TORCH_CHECK(
        is_nntile_device(weight.device()),
        "nntile embedding: weight must be on device nntile");
    TORCH_CHECK(
        is_nntile_device(indices.device()),
        "nntile embedding: indices must be on device nntile");
    TORCH_CHECK(
        indices.scalar_type() == at::ScalarType::Long,
        "nntile embedding: indices must be int64");
    TORCH_CHECK(weight.dim() == 2, "nntile embedding: weight must be 2D");
    check_fp32_contiguous_nntile(weight, "embedding");
    TORCH_CHECK(
        indices.is_contiguous(),
        "nntile embedding requires contiguous indices");

    const std::vector<int64_t> out_shape =
        embedding_output_shape(indices.sizes(), weight.size(1));
    at::Tensor output = at::empty(
        out_shape,
        weight.options().memory_format(at::MemoryFormat::Contiguous));
    const nntile::Index axis =
        static_cast<nntile::Index>(indices.dim());
    classic_tensor_embedding_forward_fp32(indices, weight, output, axis);
    return output;
}

at::Tensor embedding_backward(
    const at::Tensor &grad_output,
    const at::Tensor &indices,
    int64_t num_weights)
{
    nntile::GraphFillScope record;
    check_fp32_contiguous_nntile(grad_output, "embedding_backward");
    TORCH_CHECK(
        indices.scalar_type() == at::ScalarType::Long,
        "nntile embedding_backward: indices must be int64");
    TORCH_CHECK(
        is_nntile_device(indices.device()),
        "nntile embedding_backward: indices must be on device nntile");
    TORCH_CHECK(
        indices.is_contiguous(),
        "nntile embedding_backward requires contiguous indices");
    TORCH_CHECK(
        num_weights > 0,
        "nntile embedding_backward: invalid num_weights");

    const int64_t embed_dim = grad_output.size(-1);
    at::Tensor grad_weight = at::zeros(
        {num_weights, embed_dim},
        grad_output.options().memory_format(at::MemoryFormat::Contiguous));
    const nntile::Index axis =
        static_cast<nntile::Index>(indices.dim());
    classic_tensor_embedding_backward_fp32(
        indices,
        grad_output,
        grad_weight,
        axis,
        0);
    return grad_weight;
}

at::Tensor scale_slice_forward(
    const at::Tensor &input,
    int64_t axis,
    int64_t axis_size,
    double alpha)
{
    nntile::GraphFillScope record;
    check_fp32_contiguous_nntile(input, "scale_slice");
    int64_t const wrapped = at::maybe_wrap_dim(
        axis,
        static_cast<int64_t>(input.dim()) + 1);
    TORCH_CHECK(axis_size > 0, "nntile scale_slice: axis_size must be > 0");
    std::vector<int64_t> out_sizes(input.sizes().begin(), input.sizes().end());
    out_sizes.insert(
        out_sizes.begin() + wrapped,
        axis_size);
    at::Tensor out = at::empty(
        out_sizes,
        input.options().memory_format(at::MemoryFormat::Contiguous));
    classic_tensor_scale_slice_fp32(
        static_cast<float>(alpha),
        input,
        out,
        wrapped);
    return out;
}

at::Tensor scale_slice_backward(
    const at::Tensor &grad_out,
    int64_t axis,
    double alpha)
{
    nntile::GraphFillScope record;
    check_fp32_contiguous_nntile(grad_out, "scale_slice_backward");
    int64_t const wrapped = at::maybe_wrap_dim(axis, grad_out.dim());
    std::vector<int64_t> in_sizes(
        grad_out.sizes().begin(),
        grad_out.sizes().end());
    in_sizes.erase(in_sizes.begin() + wrapped);
    at::Tensor grad_in = at::empty(
        in_sizes,
        grad_out.options().memory_format(at::MemoryFormat::Contiguous));
    classic_tensor_sum_slice_fp32(
        grad_out,
        grad_in,
        wrapped,
        static_cast<float>(alpha),
        0.0f);
    return grad_in;
}

at::Tensor cat_forward(
    const at::Tensor &a,
    const at::Tensor &b,
    int64_t dim)
{
    nntile::GraphFillScope record;
    check_fp32_contiguous_nntile(a, "cat");
    check_fp32_contiguous_nntile(b, "cat");
    int64_t const wrapped = at::maybe_wrap_dim(dim, a.dim());
    std::vector<int64_t> out_sizes(a.sizes().begin(), a.sizes().end());
    out_sizes[static_cast<std::size_t>(wrapped)] =
        a.size(wrapped) + b.size(wrapped);
    at::Tensor out = at::empty(
        out_sizes,
        a.options().memory_format(at::MemoryFormat::Contiguous));
    classic_tensor_cat_fp32({a, b}, out, wrapped);
    return out;
}

std::tuple<at::Tensor, at::Tensor> cat_backward(
    const at::Tensor &grad_out,
    int64_t dim,
    int64_t a_size,
    int64_t b_size,
    std::array<bool, 2> output_mask)
{
    nntile::GraphFillScope record;
    check_fp32_contiguous_nntile(grad_out, "cat_backward");
    at::Tensor grad_a;
    at::Tensor grad_b;
    if (output_mask[0])
    {
        grad_a = nn_classic::narrow_forward(
            grad_out,
            dim,
            0,
            a_size);
    }
    if (output_mask[1])
    {
        grad_b = nn_classic::narrow_forward(
            grad_out,
            dim,
            a_size,
            b_size);
    }
    return {grad_a, grad_b};
}

at::Tensor narrow_forward(
    const at::Tensor &input,
    int64_t dim,
    int64_t start,
    int64_t length)
{
    nntile::GraphFillScope record;
    check_fp32_contiguous_nntile(input, "narrow");
    int64_t const wrapped = at::maybe_wrap_dim(dim, input.dim());
    TORCH_CHECK(
        start >= 0 && length >= 0 &&
            start + length <= input.size(wrapped),
        "nntile narrow: slice out of range");
    std::vector<int64_t> out_sizes(input.sizes().begin(), input.sizes().end());
    out_sizes[static_cast<std::size_t>(wrapped)] = length;
    at::Tensor out = at::empty(
        out_sizes,
        input.options().memory_format(at::MemoryFormat::Contiguous));
    classic_tensor_narrow_fp32(input, wrapped, start, length, out);
    return out;
}

at::Tensor narrow_backward(
    const at::Tensor &grad_out,
    at::IntArrayRef input_sizes,
    int64_t dim,
    int64_t start)
{
    nntile::GraphFillScope record;
    check_fp32_contiguous_nntile(grad_out, "narrow_backward");
    at::Tensor grad_in = at::empty(
        input_sizes,
        grad_out.options().memory_format(at::MemoryFormat::Contiguous));
    classic_tensor_fill_fp32(grad_in, 0.0f);
    classic_tensor_scatter_slice_fp32(
        grad_out,
        grad_in,
        dim,
        start);
    return grad_in;
}

LayerNormImpl::LayerNormImpl(int64_t normalized_size, double eps_) :
    normalized_shape({normalized_size}),
    eps(eps_)
{
    TORCH_CHECK(
        normalized_size > 0,
        "nntile LayerNorm: normalized_size must be > 0");
    weight = register_parameter(
        "weight",
        torch::ones({normalized_size}));
    bias = register_parameter(
        "bias",
        torch::zeros({normalized_size}));
}

torch::Tensor LayerNormImpl::forward(torch::Tensor x)
{
    return torch_nntile::nn_classic::layer_norm(
        x,
        normalized_shape,
        weight,
        bias,
        eps);
}

EmbeddingImpl::EmbeddingImpl(
    int64_t num_embeddings,
    int64_t embedding_dim)
{
    TORCH_CHECK(
        num_embeddings > 0 && embedding_dim > 0,
        "nntile Embedding: sizes must be > 0");
    weight = register_parameter(
        "weight",
        torch::empty({num_embeddings, embedding_dim}));
    torch::nn::init::normal_(weight);
}

torch::Tensor EmbeddingImpl::forward(torch::Tensor indices)
{
    return torch_nntile::nn_classic::nn_embedding(weight, indices);
}

} // namespace nn_classic
} // namespace torch_nntile
