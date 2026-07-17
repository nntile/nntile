/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file include/nntile/core/torch_add.hh
 * Torch-based core add (no nntile::kernel counterpart).
 *
 * @version 1.1.0
 */

#pragma once

#include <nntile/defs.h>

#ifndef NNTILE_TORCH_NATIVE_OPS
#error "nntile/core/torch_add.hh requires NNTILE_TORCH_NATIVE_OPS"
#endif

#include <nntile/base_types.hh>
#include <nntile/core/tile.hh>
#include <nntile/core/torch_meta.hh>

namespace nntile::core
{

//! Torch-based out-of-place add: out = self + alpha * other.
//! Passes Tile handles and meta to ``nntile::starpu::torch_add``.
template<typename T>
void torch_add_out(
    int starpu_worker_hint,
    const Tile<T> &self,
    const TorchTileMeta &self_meta,
    const Tile<T> &other,
    const TorchTileMeta &other_meta,
    const Tile<T> &out,
    const TorchTileMeta &out_meta,
    Scalar alpha
);

template<typename T>
void torch_add_out_async(
    int starpu_worker_hint,
    const Tile<T> &self,
    const TorchTileMeta &self_meta,
    const Tile<T> &other,
    const TorchTileMeta &other_meta,
    const Tile<T> &out,
    const TorchTileMeta &out_meta,
    Scalar alpha
);

} // namespace nntile::core
