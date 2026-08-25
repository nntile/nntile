/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_sdpa.cpp
 */

#include "nntile_sdpa.h"

#include "nntile_executor.h"
#include "nntile_graph_recorder_impl.h"
#include "nntile_layout_checks.h"
#include "nntile_tensor_gc.h"

#include <ATen/Functions.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/LegacyTypeDispatch.h>

#include <cmath>

namespace torch_nntile
{

namespace
{

bool is_nntile_device(c10::Device device)
{
    return device.type() == c10::DeviceType::PrivateUse1;
}

void check_sdpa_tensor(
    const at::Tensor &tensor,
    const char *name)
{
    TORCH_CHECK(
        is_nntile_device(tensor.device()),
        "nntile sdpa: expected nntile ",
        name);
    TORCH_CHECK(
        tensor.scalar_type() == at::ScalarType::Float,
        "nntile sdpa supports float32 only");
    TORCH_CHECK(tensor.dim() >= 3, "nntile sdpa: tensor rank must be >= 3");
}

void check_sdpa_qkv(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    int64_t batch_ndim)
{
    check_sdpa_tensor(q, "q");
    check_sdpa_tensor(k, "k");
    check_sdpa_tensor(v, "v");
    TORCH_CHECK(
        q.sizes() == k.sizes() && q.sizes() == v.sizes(),
        "nntile sdpa: Q, K, V must have the same shape");
    TORCH_CHECK(
        batch_ndim >= 1 && batch_ndim <= q.dim() - 2,
        "nntile sdpa: invalid batch_ndim");
}

void check_sdpa_mask(
    const at::Tensor &mask,
    const at::Tensor &q,
    const at::Tensor &k)
{
    TORCH_CHECK(
        mask.scalar_type() == at::ScalarType::Bool ||
            mask.scalar_type() == at::ScalarType::Byte,
        "nntile sdpa: mask must be bool or uint8");
    TORCH_CHECK(
        mask.is_contiguous(),
        "nntile sdpa: mask must be contiguous");
    TORCH_CHECK(
        is_nntile_device(mask.device()),
        "nntile sdpa: mask must be on device nntile");
    const int64_t q_ndim = q.dim();
    const int64_t k_ndim = k.dim();
    const int64_t q_seq = q.size(q_ndim - 2);
    const int64_t k_seq = k.size(k_ndim - 2);
    TORCH_CHECK(
        mask.dim() == 2 &&
            mask.size(0) == q_seq &&
            mask.size(1) == k_seq,
        "nntile sdpa: mask shape must be [q_seq, k_seq]");
}

//! Prepare attention mask for graph SDPA (BOOL on nntile).
at::Tensor mask_for_nntile_sdpa(const at::Tensor &mask)
{
    TORCH_CHECK(
        mask.scalar_type() == at::ScalarType::Bool ||
            mask.scalar_type() == at::ScalarType::Byte,
        "nntile sdpa: mask must be bool or uint8");
    TORCH_CHECK(
        mask.is_contiguous(),
        "nntile sdpa: mask must be contiguous");
    TORCH_CHECK(
        is_nntile_device(mask.device()),
        "nntile sdpa: mask must be on device nntile");
    return mask;
}

} // namespace

at::Tensor sdpa_forward(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const std::optional<at::Tensor> &mask,
    int64_t batch_ndim,
    bool is_causal)
{
    nntile::GraphFillScope record;
    check_sdpa_qkv(q, k, v, batch_ndim);
    if (mask.has_value())
    {
        TORCH_CHECK(
            mask->scalar_type() == at::ScalarType::Bool ||
                mask->scalar_type() == at::ScalarType::Byte,
            "nntile sdpa: mask must be bool or uint8");
        check_sdpa_mask(*mask, q, k);
    }

    at::Tensor out = empty_metadata_tensor(
        q.sizes(),
        q.scalar_type(),
        q.device());
    at::Tensor mask_u8;
    if (mask.has_value())
    {
        mask_u8 = mask_for_nntile_sdpa(*mask);
    }

    const at::Tensor *mask_ptr = nullptr;
    if (mask.has_value())
    {
        mask_ptr = &mask_u8;
    }

    tensor_sdpa_forward_fp32(
        q,
        k,
        v,
        mask_ptr,
        out,
        batch_ndim,
        is_causal);
    return out;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> sdpa_backward(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &grad_out,
    const std::optional<at::Tensor> &mask,
    int64_t batch_ndim,
    bool is_causal)
{
    nntile::GraphFillScope record;
    check_sdpa_qkv(q, k, v, batch_ndim);
    check_sdpa_tensor(grad_out, "grad_out");
    TORCH_CHECK(
        grad_out.sizes() == q.sizes(),
        "nntile sdpa_backward: grad_out shape must match Q");
    if (mask.has_value())
    {
        TORCH_CHECK(
            mask->scalar_type() == at::ScalarType::Bool ||
                mask->scalar_type() == at::ScalarType::Byte,
            "nntile sdpa: mask must be bool or uint8");
        check_sdpa_mask(*mask, q, k);
    }

    at::Tensor grad_q = empty_metadata_tensor(
        q.sizes(), q.scalar_type(), q.device());
    at::Tensor grad_k = empty_metadata_tensor(
        k.sizes(), k.scalar_type(), k.device());
    at::Tensor grad_v = empty_metadata_tensor(
        v.sizes(), v.scalar_type(), v.device());

    at::Tensor mask_u8;
    if (mask.has_value())
    {
        mask_u8 = mask_for_nntile_sdpa(*mask);
    }

    const at::Tensor *mask_ptr = nullptr;
    if (mask.has_value())
    {
        mask_ptr = &mask_u8;
    }

    tensor_sdpa_backward_fp32(
        q,
        k,
        v,
        mask_ptr,
        grad_out,
        grad_q,
        grad_k,
        grad_v,
        batch_ndim,
        is_causal);
    return {grad_q, grad_k, grad_v};
}

} // namespace torch_nntile
