/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_sdpa_aten.cpp
 */

#include "nntile_sdpa_aten.h"

#include "nntile_sdpa.h"

#include <ATen/SDPBackend.h>
#include <ATen/native/transformers/attention.h>
#include <torch/library.h>

#include <cmath>
#include <limits>
#include <vector>

namespace torch_nntile
{

namespace
{

bool is_nntile_device(c10::Device device)
{
    return device.type() == c10::DeviceType::PrivateUse1;
}

bool scale_is_default(
    const at::Tensor &query,
    const std::optional<double> &scale)
{
    if (!scale.has_value())
    {
        return true;
    }
    const double head_size = static_cast<double>(query.size(-1));
    const double expected = 1.0 / std::sqrt(head_size);
    return std::abs(scale.value() - expected) < 1e-6;
}

bool sdpa_inputs_supported(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    double dropout_p,
    bool enable_gqa,
    const std::optional<double> &scale)
{
    if (!is_nntile_device(query.device()) ||
        !is_nntile_device(key.device()) ||
        !is_nntile_device(value.device()))
    {
        return false;
    }
    if (query.scalar_type() != at::ScalarType::Float ||
        key.scalar_type() != at::ScalarType::Float ||
        value.scalar_type() != at::ScalarType::Float)
    {
        return false;
    }
    if (query.dim() != 3 && query.dim() != 4)
    {
        return false;
    }
    if (key.dim() != query.dim() || value.dim() != query.dim())
    {
        return false;
    }
    if (dropout_p != 0.0 || enable_gqa)
    {
        return false;
    }
    if (!scale_is_default(query, scale))
    {
        return false;
    }
    return true;
}

at::Tensor make_causal_mask(int64_t q_seq, int64_t k_seq)
{
    const auto idx_opts = at::TensorOptions().dtype(at::kLong);
    const at::Tensor k_idx = at::arange(k_seq, idx_opts);
    const at::Tensor q_idx = at::arange(q_seq, idx_opts);
    return k_idx.unsqueeze(0) <= q_idx.unsqueeze(1);
}

at::Tensor squeeze_attn_bias_to_2d(const at::Tensor &attn_bias)
{
    at::Tensor out = attn_bias;
    while (out.dim() > 2 && out.size(0) == 1)
    {
        out = out.squeeze(0);
    }
    while (out.dim() > 2)
    {
        out = out.select(0, 0);
    }
    return out.contiguous();
}

at::Tensor float_attn_bias_to_bool(const at::Tensor &attn_bias)
{
    const at::Tensor bias = attn_bias.to(at::kFloat);
    const at::Tensor finite = at::isfinite(bias);
    const at::Tensor not_neg_inf = ~at::isneginf(bias);
    return finite & not_neg_inf;
}

at::Tensor logsumexp_placeholder(const at::Tensor &query)
{
    std::vector<int64_t> shape;
    shape.reserve(static_cast<std::size_t>(query.dim() - 1));
    for (int64_t i = 0; i < query.dim() - 1; ++i)
    {
        shape.push_back(query.size(i));
    }
    return at::empty(
        shape,
        query.options().dtype(at::ScalarType::Float));
}

std::optional<at::Tensor> convert_attn_bias_to_mask(
    const std::optional<at::Tensor> &attn_bias,
    bool is_causal,
    int64_t q_seq,
    int64_t k_seq)
{
    if (is_causal)
    {
        return make_causal_mask(q_seq, k_seq);
    }
    if (!attn_bias.has_value() || !attn_bias->defined() ||
        attn_bias->numel() == 0)
    {
        return std::nullopt;
    }

    at::Tensor bias_2d = squeeze_attn_bias_to_2d(*attn_bias);
    TORCH_CHECK(
        bias_2d.dim() == 2,
        "nntile sdpa: attn_bias must be broadcastable to [q_seq, k_seq]");
    TORCH_CHECK(
        bias_2d.size(0) == q_seq && bias_2d.size(1) == k_seq,
        "nntile sdpa: attn_bias shape must be [q_seq, k_seq] after squeeze");

    at::Tensor bool_mask;
    if (bias_2d.scalar_type() == at::ScalarType::Bool)
    {
        bool_mask = bias_2d;
    }
    else
    {
        bool_mask = float_attn_bias_to_bool(bias_2d);
    }
    return bool_mask.contiguous();
}

at::Tensor ensure_contiguous_nntile(const at::Tensor &tensor)
{
    if (tensor.is_contiguous())
    {
        return tensor;
    }
    return tensor.contiguous();
}

} // namespace

int64_t fused_sdp_choice(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    const std::optional<at::Tensor> &attn_mask,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale,
    bool enable_gqa)
{
    (void)attn_mask;
    (void)is_causal;
    if (!sdpa_inputs_supported(
            query,
            key,
            value,
            dropout_p,
            enable_gqa,
            scale))
    {
        return static_cast<int64_t>(at::SDPBackend::error);
    }
    return static_cast<int64_t>(at::SDPBackend::overrideable);
}

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
    std::optional<double> scale)
{
    TORCH_CHECK(
        sdpa_inputs_supported(
            query,
            key,
            value,
            dropout_p,
            false,
            scale),
        "nntile sdpa: unsupported scaled_dot_product_attention arguments");
    TORCH_CHECK(
        query.is_contiguous() && key.is_contiguous() && value.is_contiguous(),
        "nntile sdpa: Q, K, V must be contiguous");

    const int64_t batch_ndim = query.dim() - 2;
    const c10::SymInt q_seq = query.sym_size(-2);
    const c10::SymInt k_seq = key.sym_size(-2);
    const std::optional<at::Tensor> mask = convert_attn_bias_to_mask(
        attn_bias,
        is_causal,
        q_seq.expect_int(),
        k_seq.expect_int());

    const at::Tensor out = sdpa_forward(
        query,
        key,
        value,
        mask,
        batch_ndim);
    const at::Tensor logsumexp = logsumexp_placeholder(query);
    // PyTorch SDPA API requires logsumexp; nntile softmax uses maxsumexp internally.
    // Backward ignores this tensor and uses maxsumexp buffers in sdpa_backward.
    const at::Tensor philox_seed =
        at::empty({}, query.options().dtype(at::ScalarType::Long));
    const at::Tensor philox_offset =
        at::empty({}, query.options().dtype(at::ScalarType::Long));

    at::Tensor debug_attn_mask;
    if (return_debug_mask)
    {
        std::vector<int64_t> debug_shape(query.sizes().begin(),
            query.sizes().end());
        debug_shape.back() = k_seq.expect_int();
        debug_attn_mask = at::empty(
            debug_shape,
            query.options().dtype(at::ScalarType::Float));
    }
    else
    {
        debug_attn_mask = at::Tensor();
    }

    return std::make_tuple(
        out,
        logsumexp,
        at::Tensor(),
        at::Tensor(),
        q_seq,
        k_seq,
        philox_seed,
        philox_offset,
        debug_attn_mask);
}

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
    std::optional<double> scale)
{
    (void)out;
    // PyTorch passes logsumexp from forward; sdpa_backward uses maxsumexp internally.
    (void)logsumexp;
    (void)cum_seq_q;
    (void)cum_seq_k;
    (void)max_q;
    (void)max_k;
    (void)philox_seed;
    (void)philox_offset;

    TORCH_CHECK(
        sdpa_inputs_supported(
            query,
            key,
            value,
            dropout_p,
            false,
            scale),
        "nntile sdpa backward: unsupported arguments");

    const int64_t batch_ndim = query.dim() - 2;
    const std::optional<at::Tensor> attn_bias_opt =
        attn_bias.defined() && attn_bias.numel() > 0
        ? std::optional<at::Tensor>(attn_bias)
        : std::nullopt;
    const std::optional<at::Tensor> mask = convert_attn_bias_to_mask(
        attn_bias_opt,
        is_causal,
        query.size(-2),
        key.size(-2));

    const at::Tensor grad_out_c = ensure_contiguous_nntile(grad_out);
    auto grad_qkv = sdpa_backward(
        query,
        key,
        value,
        grad_out_c,
        mask,
        batch_ndim);

    at::Tensor grad_q = grad_input_mask[0] ? std::get<0>(grad_qkv)
                                          : at::Tensor();
    at::Tensor grad_k = grad_input_mask[1] ? std::get<1>(grad_qkv)
                                          : at::Tensor();
    at::Tensor grad_v = grad_input_mask[2] ? std::get<2>(grad_qkv)
                                          : at::Tensor();

    at::Tensor grad_attn_bias;
    if (grad_input_mask[3] && attn_bias.defined() && attn_bias.numel() > 0)
    {
        grad_attn_bias = at::zeros_like(attn_bias);
    }
    else
    {
        grad_attn_bias = at::Tensor();
    }

    return std::make_tuple(grad_q, grad_k, grad_v, grad_attn_bias);
}

} // namespace torch_nntile

namespace
{

int64_t nntile_fused_sdp_choice_stub(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    const std::optional<at::Tensor> &attn_mask,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale,
    bool enable_gqa)
{
    return torch_nntile::fused_sdp_choice(
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
        scale,
        enable_gqa);
}

struct NntileFusedSdpChoiceRegistrar
{
    NntileFusedSdpChoiceRegistrar()
    {
        at::native::_fused_sdp_choice_stub.set_privateuse1_dispatch_ptr(
            &nntile_fused_sdp_choice_stub);
    }
};

const NntileFusedSdpChoiceRegistrar kNntileFusedSdpChoiceRegistrar;

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl("_fused_sdp_choice", TORCH_FN(torch_nntile::fused_sdp_choice));
    m.impl(
        "_scaled_dot_product_fused_attention_overrideable",
        TORCH_FN(torch_nntile::sdpa_overrideable_forward));
    m.impl(
        "_scaled_dot_product_fused_attention_overrideable_backward",
        TORCH_FN(torch_nntile::sdpa_overrideable_backward));
}
