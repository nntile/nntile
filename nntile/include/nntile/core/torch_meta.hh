/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file include/nntile/core/torch_meta.hh
 * Shared tile meta for torch-native StarPU codelets.
 *
 * @version 1.1.0
 */

#pragma once

#include <nntile/defs.h>

#ifndef NNTILE_TORCH_NATIVE_OPS
#error "nntile/core/torch_meta.hh requires NNTILE_TORCH_NATIVE_OPS"
#endif

#include <vector>

#include <nntile/base_types.hh>

namespace nntile::core
{

//! Tile meta for torch-native StarPU codelets (sizes, strides, offset).
//!
//! Untiled path may pass non-contiguous strides and a non-zero
//! ``storage_offset`` (element units) so HF ``transpose`` / ``narrow`` /
//! ``split`` stay zero-copy views into the parent StarPU buffer.
struct TorchTileMeta
{
    std::vector<Index> sizes;
    std::vector<Index> strides;
    //! Element offset into the StarPU buffer (not bytes).
    Index storage_offset = 0;
};

//! Build contiguous row-major strides for ``sizes`` (offset 0).
TorchTileMeta make_contiguous_torch_meta(const std::vector<Index> &sizes);

//! Max rank packed into torch-native StarPU args.
inline constexpr Index torch_native_max_ndim = 8;

} // namespace nntile::core
