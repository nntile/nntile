/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_batch_norm.cpp
 */

#include "nntile_executor.h"
#include "nntile_tensor_gc.h"

#include <ATen/Functions.h>
#include <ATen/TensorUtils.h>
#include <torch/library.h>

#include <array>

namespace torch_nntile
{

namespace
{

bool is_nntile_device(c10::Device device)
{
    return device.type() == c10::DeviceType::PrivateUse1;
}

void check_fp32(const at::Tensor &tensor, const char *name)
{
    TORCH_CHECK(is_nntile_device(tensor.device()), name, ": expected nntile");
    TORCH_CHECK(tensor.scalar_type() == at::ScalarType::Float, name, ": fp32");
}

at::Tensor as_contiguous_fp32(const at::Tensor &tensor, const char *name)
{
    check_fp32(tensor, name);
    return tensor.is_contiguous() ? tensor : tensor.contiguous();
}

void check_optional(const std::optional<at::Tensor> &tensor, const char *name)
{
    if (tensor.has_value() && tensor->defined())
    {
        check_fp32(*tensor, name);
    }
}

at::Tensor optional_contiguous(
    const std::optional<at::Tensor> &tensor,
    const char *name)
{
    if (tensor.has_value() && tensor->defined())
    {
        return as_contiguous_fp32(*tensor, name);
    }
    return at::Tensor();
}

at::Tensor const *optional_defined_ptr(const at::Tensor &tensor)
{
    return tensor.defined() ? &tensor : nullptr;
}

int64_t channel_count(const at::Tensor &input)
{
    TORCH_CHECK(input.dim() >= 2, "nntile batch_norm: input rank < 2");
    return input.size(1);
}

} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_batch_norm(
    const at::Tensor &input,
    const std::optional<at::Tensor> &weight,
    const std::optional<at::Tensor> &bias,
    const std::optional<at::Tensor> &running_mean,
    const std::optional<at::Tensor> &running_var,
    bool training,
    double momentum,
    double eps)
{
    const at::Tensor input_c =
        as_contiguous_fp32(input, "nntile native_batch_norm input");
    const at::Tensor weight_c = optional_contiguous(
        weight,
        "nntile native_batch_norm weight");
    const at::Tensor bias_c = optional_contiguous(
        bias,
        "nntile native_batch_norm bias");
    const at::Tensor running_mean_c = optional_contiguous(
        running_mean,
        "nntile native_batch_norm running_mean");
    const at::Tensor running_var_c = optional_contiguous(
        running_var,
        "nntile native_batch_norm running_var");
    const int64_t channels = channel_count(input_c);
    at::Tensor out = empty_metadata_tensor(
        input_c.sizes(),
        input_c.scalar_type(),
        input_c.device());
    const int64_t stat_size = training ? channels : 0;
    at::Tensor save_mean = empty_metadata_tensor(
        {stat_size},
        input_c.scalar_type(),
        input_c.device());
    at::Tensor save_invstd = empty_metadata_tensor(
        {stat_size},
        input_c.scalar_type(),
        input_c.device());
    tensor_native_batch_norm_fp32(
        input_c,
        optional_defined_ptr(weight_c),
        optional_defined_ptr(bias_c),
        optional_defined_ptr(running_mean_c),
        optional_defined_ptr(running_var_c),
        out,
        save_mean,
        save_invstd,
        training,
        momentum,
        eps);
    return {out, save_mean, save_invstd};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_batch_norm_backward(
    const at::Tensor &grad_out,
    const at::Tensor &input,
    const std::optional<at::Tensor> &weight,
    const std::optional<at::Tensor> &running_mean,
    const std::optional<at::Tensor> &running_var,
    const std::optional<at::Tensor> &save_mean,
    const std::optional<at::Tensor> &save_invstd,
    bool training,
    double eps,
    std::array<bool, 3> output_mask)
{
    const at::Tensor grad_c = as_contiguous_fp32(
        grad_out,
        "nntile native_batch_norm_backward grad");
    const at::Tensor input_c = as_contiguous_fp32(
        input,
        "nntile native_batch_norm_backward input");
    const at::Tensor weight_c = optional_contiguous(
        weight,
        "nntile native_batch_norm_backward weight");
    const at::Tensor running_mean_c = optional_contiguous(
        running_mean,
        "native_batch_norm_backward running_mean");
    const at::Tensor running_var_c = optional_contiguous(
        running_var,
        "native_batch_norm_backward running_var");
    const at::Tensor save_mean_c = optional_contiguous(
        save_mean,
        "native_batch_norm_backward save_mean");
    const at::Tensor save_invstd_c = optional_contiguous(
        save_invstd,
        "native_batch_norm_backward save_invstd");
    const int64_t channels = channel_count(input_c);
    at::Tensor grad_input;
    at::Tensor grad_weight;
    at::Tensor grad_bias;
    if (output_mask[0])
    {
        grad_input = empty_metadata_tensor(
            input_c.sizes(),
            input_c.scalar_type(),
            input_c.device());
    }
    if (output_mask[1])
    {
        grad_weight = empty_metadata_tensor(
            {channels},
            input_c.scalar_type(),
            input_c.device());
    }
    if (output_mask[2])
    {
        grad_bias = empty_metadata_tensor(
            {channels},
            input_c.scalar_type(),
            input_c.device());
    }
    tensor_native_batch_norm_backward_fp32(
        grad_c,
        input_c,
        optional_defined_ptr(weight_c),
        optional_defined_ptr(running_mean_c),
        optional_defined_ptr(running_var_c),
        optional_defined_ptr(save_mean_c),
        optional_defined_ptr(save_invstd_c),
        output_mask[0] ? &grad_input : nullptr,
        output_mask[1] ? &grad_weight : nullptr,
        output_mask[2] ? &grad_bias : nullptr,
        training,
        eps,
        output_mask);
    return {grad_input, grad_weight, grad_bias};
}

} // namespace torch_nntile

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl(
        "native_batch_norm",
        TORCH_FN(torch_nntile::native_batch_norm));
    m.impl(
        "native_batch_norm_backward",
        TORCH_FN(torch_nntile::native_batch_norm_backward));
}
