/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_sdpa_aten.h
 */

#pragma once

#include <ATen/ATen.h>

#include <array>
#include <cstdint>
#include <optional>
#include <tuple>

namespace torch_nntile
{

int64_t fused_sdp_choice(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    const std::optional<at::Tensor> &attn_mask,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale,
    bool enable_gqa);

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    c10::SymInt,
    c10::SymInt,
    at::Tensor,
    at::Tensor,
    at::Tensor>
sdpa_overrideable_forward(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    const std::optional<at::Tensor> &attn_bias,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    std::optional<double> scale);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
sdpa_overrideable_backward(
    const at::Tensor &grad_out,
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    const at::Tensor &attn_bias,
    std::array<bool, 4> grad_input_mask,
    const at::Tensor &out,
    const at::Tensor &logsumexp,
    const at::Tensor &cum_seq_q,
    const at::Tensor &cum_seq_k,
    c10::SymInt max_q,
    c10::SymInt max_k,
    double dropout_p,
    bool is_causal,
    const at::Tensor &philox_seed,
    const at::Tensor &philox_offset,
    std::optional<double> scale);

} // namespace torch_nntile
