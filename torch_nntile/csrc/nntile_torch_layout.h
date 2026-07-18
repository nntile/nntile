/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_torch_layout.h
 * Pack at::Tensor sizes/strides/offset into TorchDispatchArgs.
 */

#pragma once

#include <ATen/ATen.h>

#include <nntile/core/torch_dispatch.hh>
#include <nntile/core/torch_meta.hh>
#include <nntile/starpu/torch_dispatch.hh>

namespace torch_nntile
{

//! Build TorchTileMeta from a PrivateUse1 tensor (view-aware).
inline nntile::core::TorchTileMeta torch_meta_from_tensor(
    const at::Tensor &tensor)
{
    nntile::core::TorchTileMeta meta;
    const auto sizes = tensor.sizes();
    const auto strides = tensor.strides();
    meta.sizes.resize(static_cast<std::size_t>(sizes.size()));
    meta.strides.resize(static_cast<std::size_t>(strides.size()));
    for (int64_t i = 0; i < sizes.size(); ++i)
    {
        meta.sizes[static_cast<std::size_t>(i)] =
            static_cast<nntile::Index>(sizes[i]);
        meta.strides[static_cast<std::size_t>(i)] =
            static_cast<nntile::Index>(strides[i]);
    }
    meta.storage_offset =
        static_cast<nntile::Index>(tensor.storage_offset());
    return meta;
}

inline void pack_tensor_layout(
    nntile::starpu::TorchDispatchArgs &args,
    nntile::Index slot,
    const at::Tensor &tensor,
    bool is_out)
{
    nntile::core::pack_meta_into(
        args,
        slot,
        torch_meta_from_tensor(tensor),
        is_out);
}

} // namespace torch_nntile
