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
        for (Index i = 0; i < ndim; ++i)
        {
            args.in_sizes[slot][i] = meta.sizes[static_cast<size_t>(i)];
            args.in_strides[slot][i] =
                meta.strides[static_cast<size_t>(i)];
        }
    }
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

} // namespace nntile::core
