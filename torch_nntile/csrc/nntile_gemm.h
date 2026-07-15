/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_gemm.h
 */

#pragma once

#include <ATen/ATen.h>

#include <cstdint>
#include <tuple>

namespace torch_nntile
{

at::Tensor gemm_forward(
    const at::Tensor &a,
    const at::Tensor &b,
    int64_t ndim,
    int64_t batch_ndim,
    bool trans_a = false,
    bool trans_b = false);

std::tuple<at::Tensor, at::Tensor> gemm_backward(
    const at::Tensor &a,
    const at::Tensor &b,
    const at::Tensor &grad_out,
    int64_t ndim,
    int64_t batch_ndim,
    std::array<bool, 2> output_mask,
    bool trans_a = false,
    bool trans_b = false);

at::Tensor matmul_nd(
    const at::Tensor &a,
    const at::Tensor &b);

} // namespace torch_nntile
