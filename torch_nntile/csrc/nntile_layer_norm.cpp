/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_layer_norm.cpp
 * AutogradPrivateUse1 ``aten::native_layer_norm``.
 *
 * Do not register PrivateUse1 ``native_layer_norm`` (inference uses
 * CompositeExplicit ``math_native_layer_norm``). VariableType would
 * still attach ``NativeLayerNormBackward0``, which has no composite
 * in core. This Autograd kernel runs the same math (reshape +
 * ``native_batch_norm`` + affine) *with* autograd so backward is the
 * suboperations, not a fused LayerNorm backward.
 */

#include "nntile_no_implicit_copy.h"

#include <nntile/tensor/graph_fill_timer.hh>

#include <ATen/Functions.h>
#include <torch/library.h>

#include <cstdint>
#include <tuple>

namespace torch_nntile
{

namespace
{

at::Tensor optional_defined(const std::optional<at::Tensor> &tensor)
{
    if (tensor.has_value() && tensor->defined())
    {
        return *tensor;
    }
    return at::Tensor();
}

} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_layer_norm(
    const at::Tensor &input,
    at::IntArrayRef normalized_shape,
    const std::optional<at::Tensor> &weight_opt,
    const std::optional<at::Tensor> &bias_opt,
    double eps)
{
    nntile::GraphFillScope record;
    require_nntile_operand(input, "native_layer_norm", "input");
    TORCH_CHECK(
        static_cast<int64_t>(normalized_shape.size()) <= input.dim(),
        "nntile native_layer_norm: normalized_shape rank");
    const int64_t axis =
        input.dim() - static_cast<int64_t>(normalized_shape.size());
    for (size_t i = 0; i < normalized_shape.size(); ++i)
    {
        TORCH_CHECK(
            input.size(axis + static_cast<int64_t>(i))
                == normalized_shape[i],
            "nntile native_layer_norm: shape mismatch");
    }
    const at::Tensor weight = optional_defined(weight_opt);
    const at::Tensor bias = optional_defined(bias_opt);
    if (weight.defined())
    {
        require_nntile_operand(weight, "native_layer_norm", "weight");
    }
    if (bias.defined())
    {
        require_nntile_operand(bias, "native_layer_norm", "bias");
    }

    int64_t m = 1;
    for (int64_t i = 0; i < axis; ++i)
    {
        m *= input.size(i);
    }
    at::Tensor input_bn = input.reshape({1, m, -1});
    auto bn = at::native_batch_norm(
        input_bn,
        /*weight=*/std::nullopt,
        /*bias=*/std::nullopt,
        /*running_mean=*/std::nullopt,
        /*running_var=*/std::nullopt,
        /*training=*/true,
        /*momentum=*/0.0,
        eps);
    at::Tensor out = std::get<0>(bn).reshape(input.sizes());
    at::Tensor mean = std::get<1>(bn);
    at::Tensor rstd = std::get<2>(bn);
    if (weight.defined() && bias.defined())
    {
        out = at::addcmul(bias, out, weight);
    }
    else if (weight.defined())
    {
        out = at::mul(out, weight);
    }
    else if (bias.defined())
    {
        out = at::add(out, bias);
    }
    return {out, mean, rstd};
}

} // namespace torch_nntile

TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m)
{
    m.impl(
        "native_layer_norm",
        TORCH_FN(torch_nntile::native_layer_norm));
}
