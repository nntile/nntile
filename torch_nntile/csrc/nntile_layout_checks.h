/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_layout_checks.h
 * Layout checks for classic NNTile kernels vs CUDA-parity aten paths.
 */

#pragma once

#include <ATen/ATen.h>
#include <c10/util/Exception.h>

namespace torch_nntile
{

//! Classic ``torch_nntile.nn.functional`` / pybind kernels: dense storage
//! from offset 0 only. Call ``tensor.contiguous()`` explicitly (autograd
//! tracked) before invoking these ops when inputs are views.
inline void require_nntile_kernel_dense(
    const at::Tensor &tensor,
    const char *name)
{
    TORCH_CHECK(
        tensor.is_contiguous() && tensor.storage_offset() == 0,
        "nntile ",
        name,
        ": expected contiguous tensor with storage_offset==0; call "
        ".contiguous() explicitly before this classic NNTile op");
}

} // namespace torch_nntile
