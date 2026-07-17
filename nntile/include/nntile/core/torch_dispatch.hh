/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file include/nntile/core/torch_dispatch.hh
 * Core submit helpers for torch-native family codelets.
 *
 * @version 1.1.0
 */

#pragma once

#include <nntile/defs.h>

#ifndef NNTILE_TORCH_NATIVE_OPS
#error "nntile/core/torch_dispatch.hh requires NNTILE_TORCH_NATIVE_OPS"
#endif

#include <vector>

#include <nntile/base_types.hh>
#include <nntile/core/tile.hh>
#include <nntile/core/torch_meta.hh>
#include <nntile/starpu/torch_dispatch.hh>

namespace nntile::core
{

void pack_meta_into(
    starpu::TorchDispatchArgs &args,
    Index slot,
    const TorchTileMeta &meta,
    bool is_out);

void torch_unary_out(
    int starpu_worker_hint,
    starpu::TorchKind kind,
    const Tile<fp32_t> &in,
    const TorchTileMeta &in_meta,
    const Tile<fp32_t> &out,
    const TorchTileMeta &out_meta,
    const starpu::TorchDispatchArgs &extra = {});

void torch_binary_out(
    int starpu_worker_hint,
    starpu::TorchKind kind,
    const Tile<fp32_t> &a,
    const TorchTileMeta &a_meta,
    const Tile<fp32_t> &b,
    const TorchTileMeta &b_meta,
    const Tile<fp32_t> &out,
    const TorchTileMeta &out_meta,
    const starpu::TorchDispatchArgs &extra = {});

void torch_ternary_out(
    int starpu_worker_hint,
    starpu::TorchKind kind,
    const Tile<fp32_t> &a,
    const TorchTileMeta &a_meta,
    const Tile<fp32_t> &b,
    const TorchTileMeta &b_meta,
    const Tile<fp32_t> &c,
    const TorchTileMeta &c_meta,
    const Tile<fp32_t> &out,
    const TorchTileMeta &out_meta,
    const starpu::TorchDispatchArgs &extra = {});

void torch_embedding_out(
    int starpu_worker_hint,
    const Tile<fp32_t> &weight,
    const TorchTileMeta &weight_meta,
    const Tile<int64_t> &indices,
    const TorchTileMeta &indices_meta,
    const Tile<fp32_t> &out,
    const TorchTileMeta &out_meta);

void torch_cat_out(
    int starpu_worker_hint,
    Index dim,
    const std::vector<const Tile<fp32_t> *> &inputs,
    const std::vector<TorchTileMeta> &input_metas,
    const Tile<fp32_t> &out,
    const TorchTileMeta &out_meta);

void torch_layer_norm_out(
    int starpu_worker_hint,
    const Tile<fp32_t> &input,
    const TorchTileMeta &input_meta,
    const Tile<fp32_t> *weight,
    const TorchTileMeta *weight_meta,
    const Tile<fp32_t> *bias,
    const TorchTileMeta *bias_meta,
    const Tile<fp32_t> &out,
    const TorchTileMeta &out_meta,
    const Tile<fp32_t> &mean,
    const TorchTileMeta &mean_meta,
    const Tile<fp32_t> &rstd,
    const TorchTileMeta &rstd_meta,
    Index normalized_ndim,
    Scalar eps);

} // namespace nntile::core
