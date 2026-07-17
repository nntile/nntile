/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file src/core/torch_add.cc
 * Torch-based core add → starpu::torch_add.submit.
 *
 * @version 1.1.0
 */

#include "nntile/core/torch_add.hh"

#include <stdexcept>

#include "nntile/starpu/config.hh"
#include "nntile/starpu/torch_add.hh"

namespace nntile::core
{

namespace
{

starpu::TorchAdd<std::tuple<fp32_t>>::args_t pack_args(
    const TorchTileMeta &self_meta,
    const TorchTileMeta &other_meta,
    const TorchTileMeta &out_meta,
    Scalar alpha
)
{
    if (self_meta.sizes != other_meta.sizes ||
        self_meta.sizes != out_meta.sizes)
    {
        throw std::runtime_error(
            "torch_add_out: size mismatch between tensors");
    }
    if (self_meta.sizes.size() != self_meta.strides.size() ||
        other_meta.sizes.size() != other_meta.strides.size() ||
        out_meta.sizes.size() != out_meta.strides.size())
    {
        throw std::runtime_error(
            "torch_add_out: strides rank mismatch");
    }
    const Index ndim = static_cast<Index>(self_meta.sizes.size());
    if (ndim > starpu::torch_add_max_ndim)
    {
        throw std::runtime_error(
            "torch_add_out: ndim exceeds torch_add_max_ndim");
    }
    starpu::TorchAdd<std::tuple<fp32_t>>::args_t args{};
    args.ndim = ndim;
    args.alpha = alpha;
    for (Index i = 0; i < ndim; ++i)
    {
        args.sizes[i] = self_meta.sizes[static_cast<size_t>(i)];
        args.self_strides[i] =
            self_meta.strides[static_cast<size_t>(i)];
        args.other_strides[i] =
            other_meta.strides[static_cast<size_t>(i)];
        args.out_strides[i] =
            out_meta.strides[static_cast<size_t>(i)];
    }
    return args;
}

} // namespace

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
)
{
    if (self.ndim != static_cast<Index>(self_meta.sizes.size()) ||
        other.ndim != static_cast<Index>(other_meta.sizes.size()) ||
        out.ndim != static_cast<Index>(out_meta.sizes.size()))
    {
        throw std::runtime_error(
            "torch_add_out: tile ndim disagrees with meta");
    }
    for (Index i = 0; i < self.ndim; ++i)
    {
        if (self.shape[i] != self_meta.sizes[static_cast<size_t>(i)] ||
            other.shape[i] != other_meta.sizes[static_cast<size_t>(i)] ||
            out.shape[i] != out_meta.sizes[static_cast<size_t>(i)])
        {
            throw std::runtime_error(
                "torch_add_out: tile shape disagrees with meta");
        }
    }

    int mpi_rank = starpu_mpi_world_rank();
    int out_rank = out.mpi_get_rank();
    self.mpi_transfer(out_rank, mpi_rank);
    other.mpi_transfer(out_rank, mpi_rank);
    if (mpi_rank != out_rank)
    {
        return;
    }

    auto args = pack_args(self_meta, other_meta, out_meta, alpha);
    starpu::torch_add.submit<std::tuple<T>>(
        starpu_worker_hint,
        args,
        self,
        other,
        out);
}

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
)
{
    torch_add_out_async<T>(
        starpu_worker_hint,
        self,
        self_meta,
        other,
        other_meta,
        out,
        out_meta,
        alpha);
    nntile::starpu_task_wait_for_all_unless_deferred();
}

template
void torch_add_out_async<fp32_t>(
    int,
    const Tile<fp32_t> &,
    const TorchTileMeta &,
    const Tile<fp32_t> &,
    const TorchTileMeta &,
    const Tile<fp32_t> &,
    const TorchTileMeta &,
    Scalar
);

template
void torch_add_out<fp32_t>(
    int,
    const Tile<fp32_t> &,
    const TorchTileMeta &,
    const Tile<fp32_t> &,
    const TorchTileMeta &,
    const Tile<fp32_t> &,
    const TorchTileMeta &,
    Scalar
);

} // namespace nntile::core
