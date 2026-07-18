/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_sdpa.cpp
 */

#include "nntile_sdpa.h"

#include "nntile_executor.h"
#include "nntile_graph_recorder_impl.h"
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

//! Densify for StarPU without nesting ContiguousFn autograd.
//!
//! ``sdpa_forward`` / ``sdpa_backward`` run under PyTorch's fused-SDPA
//! autograd wrapper (or ``SdpaKernelFn``). Calling ``tensor.contiguous()``
//! there would register ``ContiguousFn`` mid-forward and break
//! split→view→transpose→SDPA backward (Cat on uninitialized handles).
at::Tensor densify_sdpa_operand(const at::Tensor &tensor)
{
    if (tensor.is_contiguous())
    {
        return tensor;
    }
    at::AutoDispatchBelowADInplaceOrView guard;
    return tensor.contiguous();
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
    // Untiled: non-contiguous views OK (sizes/strides/offset packed).
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
//!
//! The executor records the mask as ``DataType::BOOL``. Do **not** round-trip
//! through ``mask.cpu()`` - that gathers through StarPU and syncs every
//! attention layer during graph recording.
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
    // Byte masks are accepted at the API boundary; the executor binds them
    // as BOOL logical nodes (same 1-byte element size).
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
    // Fused c_attn → split → transpose heads yields views whose last-dim
    // stride spans the full 3H packed width. CUDA SDPA / densify copy_
    // expects dense [B,H,S,D] buffers — densify at this boundary (graph
    // Copy only; no ContiguousFn nested under fused-SDPA autograd).
    const at::Tensor q_c = densify_sdpa_operand(q);
    const at::Tensor k_c = densify_sdpa_operand(k);
    const at::Tensor v_c = densify_sdpa_operand(v);
    check_sdpa_qkv(q_c, k_c, v_c, batch_ndim);
    if (mask.has_value())
    {
        TORCH_CHECK(
            mask->scalar_type() == at::ScalarType::Bool ||
                mask->scalar_type() == at::ScalarType::Byte,
            "nntile sdpa: mask must be bool or uint8");
        check_sdpa_mask(*mask, q_c, k_c);
    }

    at::Tensor out = empty_metadata_tensor(
        q_c.sizes(),
        q_c.scalar_type(),
        q_c.device());
    at::Tensor mask_u8;
    std::vector<at::Tensor> inputs = {q_c, k_c, v_c};
    if (mask.has_value())
    {
        mask_u8 = mask_for_nntile_sdpa(*mask);
        inputs.push_back(mask_u8);
    }

    const at::Tensor *mask_ptr = nullptr;
    if (mask.has_value())
    {
        mask_ptr = &mask_u8;
    }

    tensor_sdpa_forward_fp32(
        q_c,
        k_c,
        v_c,
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
    const at::Tensor q_c = densify_sdpa_operand(q);
    const at::Tensor k_c = densify_sdpa_operand(k);
    const at::Tensor v_c = densify_sdpa_operand(v);
    const at::Tensor go_c = densify_sdpa_operand(grad_out);
    check_sdpa_qkv(q_c, k_c, v_c, batch_ndim);
    check_sdpa_tensor(go_c, "grad_out");
    TORCH_CHECK(
        go_c.sizes() == q_c.sizes(),
        "nntile sdpa_backward: grad_out shape must match Q");
    if (mask.has_value())
    {
        TORCH_CHECK(
            mask->scalar_type() == at::ScalarType::Bool ||
                mask->scalar_type() == at::ScalarType::Byte,
            "nntile sdpa: mask must be bool or uint8");
        check_sdpa_mask(*mask, q_c, k_c);
    }

    at::Tensor grad_q = empty_metadata_tensor(
        q_c.sizes(), q_c.scalar_type(), q_c.device());
    at::Tensor grad_k = empty_metadata_tensor(
        k_c.sizes(), k_c.scalar_type(), k_c.device());
    at::Tensor grad_v = empty_metadata_tensor(
        v_c.sizes(), v_c.scalar_type(), v_c.device());

    at::Tensor mask_u8;
    std::vector<at::Tensor> inputs = {q_c, k_c, v_c, go_c};
    if (mask.has_value())
    {
        mask_u8 = mask_for_nntile_sdpa(*mask);
        inputs.push_back(mask_u8);
    }

    const at::Tensor *mask_ptr = nullptr;
    if (mask.has_value())
    {
        mask_ptr = &mask_u8;
    }

    tensor_sdpa_backward_fp32(
        q_c,
        k_c,
        v_c,
        mask_ptr,
        go_c,
        grad_q,
        grad_k,
        grad_v,
        batch_ndim,
        is_causal);
    return {grad_q, grad_k, grad_v};
}

} // namespace torch_nntile
