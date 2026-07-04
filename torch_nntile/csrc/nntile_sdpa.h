/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_sdpa.h
 */

#pragma once

#include <ATen/ATen.h>

#include <cstdint>
#include <tuple>

namespace torch_nntile
{

at::Tensor sdpa_forward(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const std::optional<at::Tensor> &mask,
    int64_t batch_ndim);

std::tuple<at::Tensor, at::Tensor, at::Tensor> sdpa_backward(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &grad_out,
    const std::optional<at::Tensor> &mask,
    int64_t batch_ndim);

} // namespace torch_nntile
