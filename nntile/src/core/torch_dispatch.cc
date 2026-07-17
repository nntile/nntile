/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file src/core/torch_dispatch.cc
 * Core → StarPU submit for torch-native family codelets.
 *
 * @version 1.1.0
 */

#include "nntile/core/torch_dispatch.hh"

#include <stdexcept>

#include "nntile/starpu/config.hh"

namespace nntile::core
{

void pack_meta_into(
    starpu::TorchDispatchArgs &args,
    Index slot,
    const TorchTileMeta &meta,
    bool is_out)
{
    if (meta.sizes.size() != meta.strides.size())
    {
        throw std::runtime_error("torch_dispatch: strides rank mismatch");
    }
    const Index ndim = static_cast<Index>(meta.sizes.size());
    if (ndim > starpu::torch_dispatch_max_ndim)
    {
        throw std::runtime_error("torch_dispatch: ndim too large");
    }
    if (is_out)
    {
        args.out_ndim[slot] = ndim;
        args.out_offset[slot] = meta.storage_offset;
        for (Index i = 0; i < ndim; ++i)
        {
            args.out_sizes[slot][i] = meta.sizes[static_cast<size_t>(i)];
            args.out_strides[slot][i] =
                meta.strides[static_cast<size_t>(i)];
        }
    }
    else
    {
        args.in_ndim[slot] = ndim;
        args.in_offset[slot] = meta.storage_offset;
        for (Index i = 0; i < ndim; ++i)
        {
            args.in_sizes[slot][i] = meta.sizes[static_cast<size_t>(i)];
            args.in_strides[slot][i] =
                meta.strides[static_cast<size_t>(i)];
        }
    }
}

TorchTileMeta meta_from_args_or_contiguous(
    const starpu::TorchDispatchArgs &args,
    Index slot,
    bool is_out,
    const std::vector<Index> &tile_shape)
{
    const Index ndim =
        is_out ? args.out_ndim[slot] : args.in_ndim[slot];
    if (ndim <= 0)
    {
        return make_contiguous_torch_meta(tile_shape);
    }
    TorchTileMeta meta;
    meta.sizes.resize(static_cast<size_t>(ndim));
    meta.strides.resize(static_cast<size_t>(ndim));
    meta.storage_offset =
        is_out ? args.out_offset[slot] : args.in_offset[slot];
    for (Index i = 0; i < ndim; ++i)
    {
        meta.sizes[static_cast<size_t>(i)] = is_out
            ? args.out_sizes[slot][i]
            : args.in_sizes[slot][i];
        meta.strides[static_cast<size_t>(i)] = is_out
            ? args.out_strides[slot][i]
            : args.in_strides[slot][i];
    }
    return meta;
}

void torch_unary_out(
    int starpu_worker_hint,
    starpu::TorchKind kind,
    const Tile<fp32_t> &in,
    const TorchTileMeta &in_meta,
    const Tile<fp32_t> &out,
    const TorchTileMeta &out_meta,
    const starpu::TorchDispatchArgs &extra)
{
    int mpi_rank = starpu_mpi_world_rank();
    int out_rank = out.mpi_get_rank();
    in.mpi_transfer(out_rank, mpi_rank);
    if (mpi_rank != out_rank)
    {
        return;
    }
    starpu::TorchDispatchArgs args = extra;
    args.kind = kind;
    args.n_in = 1;
    args.n_out = 1;
    pack_meta_into(args, 0, in_meta, false);
    pack_meta_into(args, 0, out_meta, true);
    starpu::torch_unary.submit<std::tuple<fp32_t>>(
        starpu_worker_hint,
        args,
        in,
        out);
}

void torch_binary_out(
    int starpu_worker_hint,
    starpu::TorchKind kind,
    const Tile<fp32_t> &a,
    const TorchTileMeta &a_meta,
    const Tile<fp32_t> &b,
    const TorchTileMeta &b_meta,
    const Tile<fp32_t> &out,
    const TorchTileMeta &out_meta,
    const starpu::TorchDispatchArgs &extra)
{
    int mpi_rank = starpu_mpi_world_rank();
    int out_rank = out.mpi_get_rank();
    a.mpi_transfer(out_rank, mpi_rank);
    b.mpi_transfer(out_rank, mpi_rank);
    if (mpi_rank != out_rank)
    {
        return;
    }
    starpu::TorchDispatchArgs args = extra;
    args.kind = kind;
    args.n_in = 2;
    args.n_out = 1;
    pack_meta_into(args, 0, a_meta, false);
    pack_meta_into(args, 1, b_meta, false);
    pack_meta_into(args, 0, out_meta, true);
    starpu::torch_binary.submit<std::tuple<fp32_t>>(
        starpu_worker_hint,
        args,
        a,
        b,
        out);
}

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
    const starpu::TorchDispatchArgs &extra)
{
    int mpi_rank = starpu_mpi_world_rank();
    int out_rank = out.mpi_get_rank();
    a.mpi_transfer(out_rank, mpi_rank);
    b.mpi_transfer(out_rank, mpi_rank);
    c.mpi_transfer(out_rank, mpi_rank);
    if (mpi_rank != out_rank)
    {
        return;
    }
    starpu::TorchDispatchArgs args = extra;
    args.kind = kind;
    args.n_in = 3;
    args.n_out = 1;
    pack_meta_into(args, 0, a_meta, false);
    pack_meta_into(args, 1, b_meta, false);
    pack_meta_into(args, 2, c_meta, false);
    pack_meta_into(args, 0, out_meta, true);
    starpu::torch_ternary.submit<std::tuple<fp32_t>>(
        starpu_worker_hint,
        args,
        a,
        b,
        c,
        out);
}

void torch_embedding_out(
    int starpu_worker_hint,
    const Tile<fp32_t> &weight,
    const TorchTileMeta &weight_meta,
    const Tile<int64_t> &indices,
    const TorchTileMeta &indices_meta,
    const Tile<fp32_t> &out,
    const TorchTileMeta &out_meta)
{
    int mpi_rank = starpu_mpi_world_rank();
    int out_rank = out.mpi_get_rank();
    weight.mpi_transfer(out_rank, mpi_rank);
    indices.mpi_transfer(out_rank, mpi_rank);
    if (mpi_rank != out_rank)
    {
        return;
    }
    starpu::TorchDispatchArgs args{};
    args.kind = starpu::TorchKind::Embedding;
    args.n_in = 2;
    args.n_out = 1;
    pack_meta_into(args, 0, weight_meta, false);
    pack_meta_into(args, 1, indices_meta, false);
    pack_meta_into(args, 0, out_meta, true);
    starpu::torch_embedding.submit(
        starpu_worker_hint,
        args,
        weight,
        indices,
        out);
}

void torch_cat_out(
    int starpu_worker_hint,
    Index dim,
    const std::vector<const Tile<fp32_t> *> &inputs,
    const std::vector<TorchTileMeta> &input_metas,
    const Tile<fp32_t> &out,
    const TorchTileMeta &out_meta)
{
    if (inputs.size() != input_metas.size() || inputs.empty())
    {
        throw std::runtime_error("torch_cat_out: bad inputs");
    }
    int mpi_rank = starpu_mpi_world_rank();
    int out_rank = out.mpi_get_rank();
    for (auto *t : inputs)
    {
        t->mpi_transfer(out_rank, mpi_rank);
    }
    if (mpi_rank != out_rank)
    {
        return;
    }
    starpu::TorchDispatchArgs args{};
    args.kind = starpu::TorchKind::Cat;
    args.n_in = static_cast<Index>(inputs.size());
    args.n_out = 1;
    args.iargs[0] = dim;
    args.iargs[1] = args.n_in;
    for (Index i = 0; i < args.n_in; ++i)
    {
        pack_meta_into(
            args,
            i,
            input_metas[static_cast<size_t>(i)],
            false);
    }
    pack_meta_into(args, 0, out_meta, true);
    std::vector<starpu::Handle> handles;
    handles.reserve(inputs.size());
    for (auto *t : inputs)
    {
        handles.push_back(*t);
    }
    starpu::torch_cat.submit(
        starpu_worker_hint,
        args,
        handles,
        out);
}

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
    Scalar eps)
{
    int mpi_rank = starpu_mpi_world_rank();
    int out_rank = out.mpi_get_rank();
    input.mpi_transfer(out_rank, mpi_rank);
    if (weight != nullptr)
    {
        weight->mpi_transfer(out_rank, mpi_rank);
    }
    if (bias != nullptr)
    {
        bias->mpi_transfer(out_rank, mpi_rank);
    }
    if (mpi_rank != out_rank)
    {
        return;
    }
    starpu::TorchDispatchArgs args{};
    args.kind = starpu::TorchKind::NativeLayerNorm;
    args.n_in = 1 + (weight != nullptr) + (bias != nullptr);
    args.n_out = 3;
    args.iargs[0] = normalized_ndim;
    args.scalars[0] = eps;
    pack_meta_into(args, 0, input_meta, false);
    if (weight != nullptr && weight_meta != nullptr)
    {
        pack_meta_into(args, 1, *weight_meta, false);
    }
    if (bias != nullptr && bias_meta != nullptr)
    {
        pack_meta_into(args, 2, *bias_meta, false);
    }
    pack_meta_into(args, 0, out_meta, true);
    pack_meta_into(args, 1, mean_meta, true);
    pack_meta_into(args, 2, rstd_meta, true);
    starpu::torch_layer_norm.submit(
        starpu_worker_hint,
        args,
        input,
        weight != nullptr ? *weight : input,
        bias != nullptr ? *bias : input,
        out,
        mean,
        rstd,
        weight != nullptr,
        bias != nullptr);
}

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
    bool need_grad_bias)
{
    int mpi_rank = starpu_mpi_world_rank();
    int out_rank = grad_out.mpi_get_rank();
    if (need_grad_input && grad_input != nullptr)
    {
        out_rank = grad_input->mpi_get_rank();
    }
    else if (need_grad_weight && grad_weight != nullptr)
    {
        out_rank = grad_weight->mpi_get_rank();
    }
    else if (need_grad_bias && grad_bias != nullptr)
    {
        out_rank = grad_bias->mpi_get_rank();
    }
    grad_out.mpi_transfer(out_rank, mpi_rank);
    input.mpi_transfer(out_rank, mpi_rank);
    mean.mpi_transfer(out_rank, mpi_rank);
    rstd.mpi_transfer(out_rank, mpi_rank);
    if (weight != nullptr)
    {
        weight->mpi_transfer(out_rank, mpi_rank);
    }
    if (bias != nullptr)
    {
        bias->mpi_transfer(out_rank, mpi_rank);
    }
    if (mpi_rank != out_rank)
    {
        return;
    }
    starpu::TorchDispatchArgs args{};
    args.kind = starpu::TorchKind::NativeLayerNormBackward;
    args.n_in = 4 + (weight != nullptr) + (bias != nullptr);
    args.n_out = static_cast<Index>(need_grad_input)
        + static_cast<Index>(need_grad_weight)
        + static_cast<Index>(need_grad_bias);
    args.iargs[0] = normalized_ndim;
    pack_meta_into(args, 0, grad_out_meta, false);
    pack_meta_into(args, 1, input_meta, false);
    pack_meta_into(args, 2, mean_meta, false);
    pack_meta_into(args, 3, rstd_meta, false);
    if (weight != nullptr && weight_meta != nullptr)
    {
        pack_meta_into(args, 4, *weight_meta, false);
    }
    if (bias != nullptr && bias_meta != nullptr)
    {
        pack_meta_into(args, 5, *bias_meta, false);
    }
    if (need_grad_input && grad_input_meta != nullptr)
    {
        pack_meta_into(args, 0, *grad_input_meta, true);
    }
    if (need_grad_weight && grad_weight_meta != nullptr)
    {
        pack_meta_into(args, 1, *grad_weight_meta, true);
    }
    if (need_grad_bias && grad_bias_meta != nullptr)
    {
        pack_meta_into(args, 2, *grad_bias_meta, true);
    }
    starpu::torch_layer_norm_backward.submit(
        starpu_worker_hint,
        args,
        grad_out,
        input,
        mean,
        rstd,
        weight != nullptr ? *weight : input,
        bias != nullptr ? *bias : input,
        grad_input != nullptr ? *grad_input : input,
        grad_weight != nullptr ? *grad_weight : input,
        grad_bias != nullptr ? *grad_bias : input,
        weight != nullptr,
        bias != nullptr,
        need_grad_input,
        need_grad_weight,
        need_grad_bias);
}

void torch_embedding_dense_backward_out(
    int starpu_worker_hint,
    const Tile<fp32_t> &grad,
    const TorchTileMeta &grad_meta,
    const Tile<int64_t> &indices,
    const TorchTileMeta &indices_meta,
    const Tile<fp32_t> &grad_weight,
    const TorchTileMeta &grad_weight_meta)
{
    int mpi_rank = starpu_mpi_world_rank();
    int out_rank = grad_weight.mpi_get_rank();
    grad.mpi_transfer(out_rank, mpi_rank);
    indices.mpi_transfer(out_rank, mpi_rank);
    if (mpi_rank != out_rank)
    {
        return;
    }
    starpu::TorchDispatchArgs args{};
    args.kind = starpu::TorchKind::EmbeddingDenseBackward;
    args.n_in = 2;
    args.n_out = 1;
    args.iargs[0] = grad_weight_meta.sizes.empty()
        ? 0
        : grad_weight_meta.sizes[0];
    pack_meta_into(args, 0, grad_meta, false);
    pack_meta_into(args, 1, indices_meta, false);
    pack_meta_into(args, 0, grad_weight_meta, true);
    starpu::torch_embedding_dense_backward.submit(
        starpu_worker_hint,
        args,
        grad,
        indices,
        grad_weight);
}

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
    bool is_causal)
{
    int mpi_rank = starpu_mpi_world_rank();
    int out_rank = grad_q.mpi_get_rank();
    q.mpi_transfer(out_rank, mpi_rank);
    k.mpi_transfer(out_rank, mpi_rank);
    v.mpi_transfer(out_rank, mpi_rank);
    grad_out.mpi_transfer(out_rank, mpi_rank);
    if (mask != nullptr)
    {
        mask->mpi_transfer(out_rank, mpi_rank);
    }
    if (mpi_rank != out_rank)
    {
        return;
    }
    starpu::TorchDispatchArgs args{};
    args.kind = starpu::TorchKind::SdpaBackward;
    args.n_in = 4 + (mask != nullptr ? 1 : 0);
    args.n_out = 3;
    args.iargs[0] = mask != nullptr ? 1 : 0;
    args.iargs[1] = is_causal ? 1 : 0;
    pack_meta_into(args, 0, q_meta, false);
    pack_meta_into(args, 1, k_meta, false);
    pack_meta_into(args, 2, v_meta, false);
    pack_meta_into(args, 3, grad_out_meta, false);
    if (mask != nullptr && mask_meta != nullptr)
    {
        pack_meta_into(args, 4, *mask_meta, false);
    }
    pack_meta_into(args, 0, grad_q_meta, true);
    pack_meta_into(args, 1, grad_k_meta, true);
    pack_meta_into(args, 2, grad_v_meta, true);
    starpu::Handle mask_handle = q;
    if (mask != nullptr)
    {
        mask_handle = *mask;
    }
    starpu::torch_sdpa_backward.submit(
        starpu_worker_hint,
        args,
        q,
        k,
        v,
        grad_out,
        mask_handle,
        grad_q,
        grad_k,
        grad_v,
        mask != nullptr);
}

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
    Index ignore_index)
{
    int mpi_rank = starpu_mpi_world_rank();
    int out_rank = loss.mpi_get_rank();
    log_probs.mpi_transfer(out_rank, mpi_rank);
    target.mpi_transfer(out_rank, mpi_rank);
    if (mpi_rank != out_rank)
    {
        return;
    }
    starpu::TorchDispatchArgs args{};
    args.kind = starpu::TorchKind::NllLossForward;
    args.n_in = 2;
    args.n_out = 2;
    args.iargs[0] = reduction;
    args.iargs[1] = ignore_index;
    pack_meta_into(args, 0, log_probs_meta, false);
    pack_meta_into(args, 1, target_meta, false);
    pack_meta_into(args, 0, loss_meta, true);
    pack_meta_into(args, 1, total_weight_meta, true);
    starpu::torch_nll_loss_forward.submit(
        starpu_worker_hint,
        args,
        log_probs,
        target,
        loss,
        total_weight);
}

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
    Index ignore_index)
{
    int mpi_rank = starpu_mpi_world_rank();
    int out_rank = grad_input.mpi_get_rank();
    grad_output.mpi_transfer(out_rank, mpi_rank);
    log_probs.mpi_transfer(out_rank, mpi_rank);
    target.mpi_transfer(out_rank, mpi_rank);
    total_weight.mpi_transfer(out_rank, mpi_rank);
    if (mpi_rank != out_rank)
    {
        return;
    }
    starpu::TorchDispatchArgs args{};
    args.kind = starpu::TorchKind::NllLossBackward;
    args.n_in = 4;
    args.n_out = 1;
    args.iargs[0] = reduction;
    args.iargs[1] = ignore_index;
    pack_meta_into(args, 0, grad_output_meta, false);
    pack_meta_into(args, 1, log_probs_meta, false);
    pack_meta_into(args, 2, target_meta, false);
    pack_meta_into(args, 3, total_weight_meta, false);
    pack_meta_into(args, 0, grad_input_meta, true);
    starpu::torch_nll_loss_backward.submit(
        starpu_worker_hint,
        args,
        grad_output,
        log_probs,
        target,
        total_weight,
        grad_input);
}

} // namespace nntile::core
