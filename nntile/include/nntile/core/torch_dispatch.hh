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

//! Unpack layout from ``args`` if ``*_layout_set`` is set.
//!
//! Otherwise fall back to contiguous ``tile_shape`` (StarPU storage
//! shape). That fallback is only valid for full-tile contiguous
//! operands. View-aware record paths must ``pack_meta_into`` /
//! ``pack_tensor_layout`` from the ``at::Tensor`` (sizes, strides,
//! storage_offset) so execute-time ``from_blob`` rebuilds the same view.
//! Packed scalars use ``ndim == 0`` with ``layout_set``; do not treat
//! ``ndim == 0`` alone as unpacked.
TorchTileMeta meta_from_args_or_contiguous(
    const starpu::TorchDispatchArgs &args,
    Index slot,
    bool is_out,
    const std::vector<Index> &tile_shape);

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

void torch_layer_norm_backward_out(
    int starpu_worker_hint,
    const Tile<fp32_t> &grad_out,
    const TorchTileMeta &grad_out_meta,
    const Tile<fp32_t> &input,
    const TorchTileMeta &input_meta,
    const Tile<fp32_t> &mean,
    const TorchTileMeta &mean_meta,
    const Tile<fp32_t> &rstd,
    const TorchTileMeta &rstd_meta,
    const Tile<fp32_t> *weight,
    const TorchTileMeta *weight_meta,
    const Tile<fp32_t> *bias,
    const TorchTileMeta *bias_meta,
    const Tile<fp32_t> *grad_input,
    const TorchTileMeta *grad_input_meta,
    const Tile<fp32_t> *grad_weight,
    const TorchTileMeta *grad_weight_meta,
    const Tile<fp32_t> *grad_bias,
    const TorchTileMeta *grad_bias_meta,
    Index normalized_ndim,
    bool need_grad_input,
    bool need_grad_weight,
    bool need_grad_bias);

void torch_embedding_dense_backward_out(
    int starpu_worker_hint,
    const Tile<fp32_t> &grad,
    const TorchTileMeta &grad_meta,
    const Tile<int64_t> &indices,
    const TorchTileMeta &indices_meta,
    const Tile<fp32_t> &grad_weight,
    const TorchTileMeta &grad_weight_meta);

void torch_sdpa_backward_out(
    int starpu_worker_hint,
    const Tile<fp32_t> &q,
    const TorchTileMeta &q_meta,
    const Tile<fp32_t> &k,
    const TorchTileMeta &k_meta,
    const Tile<fp32_t> &v,
    const TorchTileMeta &v_meta,
    const Tile<fp32_t> &grad_out,
    const TorchTileMeta &grad_out_meta,
    const Tile<bool_t> *mask,
    const TorchTileMeta *mask_meta,
    const Tile<fp32_t> &grad_q,
    const TorchTileMeta &grad_q_meta,
    const Tile<fp32_t> &grad_k,
    const TorchTileMeta &grad_k_meta,
    const Tile<fp32_t> &grad_v,
    const TorchTileMeta &grad_v_meta,
    bool is_causal);

void torch_nll_loss_forward_out(
    int starpu_worker_hint,
    const Tile<fp32_t> &log_probs,
    const TorchTileMeta &log_probs_meta,
    const Tile<int64_t> &target,
    const TorchTileMeta &target_meta,
    const Tile<fp32_t> &loss,
    const TorchTileMeta &loss_meta,
    const Tile<fp32_t> &total_weight,
    const TorchTileMeta &total_weight_meta,
    Index reduction,
    Index ignore_index);

void torch_nll_loss_backward_out(
    int starpu_worker_hint,
    const Tile<fp32_t> &grad_output,
    const TorchTileMeta &grad_output_meta,
    const Tile<fp32_t> &log_probs,
    const TorchTileMeta &log_probs_meta,
    const Tile<int64_t> &target,
    const TorchTileMeta &target_meta,
    const Tile<fp32_t> &total_weight,
    const TorchTileMeta &total_weight_meta,
    const Tile<fp32_t> &grad_input,
    const TorchTileMeta &grad_input_meta,
    Index reduction,
    Index ignore_index);

} // namespace nntile::core
