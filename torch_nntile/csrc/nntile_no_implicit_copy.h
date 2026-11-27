/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_no_implicit_copy.h
 * Reject silent nntile ↔ CPU copies in compute kernels.
 */

#pragma once

#include <ATen/Tensor.h>
#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>

namespace torch_nntile
{

inline bool is_nntile_device(c10::Device device)
{
    return device.type() == c10::DeviceType::PrivateUse1;
}

inline bool is_cpu_scalar_tensor(const at::Tensor &t)
{
    return t.is_cpu() && t.numel() == 1;
}

//! Compute kernels must not gather/scatter. User moves with ``.to()``.
inline void require_nntile_operand(
    const at::Tensor &tensor,
    const char *op,
    const char *role)
{
    TORCH_CHECK(
        is_nntile_device(tensor.device()),
        "torch_nntile ",
        op,
        ": ",
        role,
        " is on ",
        tensor.device(),
        "; implicit nntile<->CPU copies are disabled. "
        "Move with .to(\"nntile\") or .to(\"cpu\") explicitly.");
}

} // namespace torch_nntile
