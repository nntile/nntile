/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_norm.h
 */

#pragma once

#include <ATen/Tensor.h>
#include <cstdint>
#include <optional>
#include <tuple>

namespace torch_nntile
{

std::tuple<at::Tensor, at::Tensor> norm_forward(
    const at::Tensor &input,
    std::optional<int64_t> dim,
    bool keepdim);

at::Tensor norm_backward(
    const at::Tensor &grad_out,
    const at::Tensor &input,
    const at::Tensor &norm_values,
    std::optional<int64_t> dim,
    bool keepdim);

at::Tensor linalg_vector_norm_nntile(
    const at::Tensor &self,
    const at::Scalar &ord,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    std::optional<at::ScalarType> dtype);

} // namespace torch_nntile
