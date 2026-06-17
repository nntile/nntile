/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/core/logsumexp.cc
 * Max and sum of exponents of Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/core/logsumexp.hh"
#include "nntile/starpu/logsumexp.hh"
#include "nntile/starpu/config.hh"

namespace nntile::core
{

template<typename T>
void logsumexp_async(int starpu_worker_hint, const Tile<T> &src, const Tile<T> &dst)
// TODO - add description
{
    // Check dimensions
    if(src.ndim - 1 != dst.ndim)
    {
        throw std::runtime_error("src.ndim - 1 != dst.ndim");
    }
    Index ndim = src.ndim;
    // Treat special case of ndim=0
    if(ndim == 0)
    {
        throw std::runtime_error("Scalar input makes no sense");
    }
    if(src.shape[src.ndim-1] != 2)
    {
        throw std::runtime_error("src last dim must be 2");
    }
    for(Index i = 0; i < ndim - 1; ++i)
    {
        if (src.shape[i] != dst.shape[i])
        {
            throw std::runtime_error("src.shape[i] != dst.shape[i]");
        }
    }
    int mpi_rank = starpu_mpi_world_rank();
    int dst_rank = dst.mpi_get_rank();
    src.mpi_transfer(dst_rank, mpi_rank);
    if(mpi_rank == dst_rank)
    {
        // Insert task
        starpu::logsumexp.submit<std::tuple<T>>(starpu_worker_hint, dst.nelems, src, dst);
    }
}

//! Tile-wise logsumexp
template<typename T>
void logsumexp(int starpu_worker_hint, const Tile<T> &src, const Tile<T> &dst)
{
    logsumexp_async<T>(starpu_worker_hint, src, dst);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void logsumexp_async<fp32_t>(int starpu_worker_hint, const Tile<fp32_t> &src, const Tile<fp32_t> &dst);

template
void logsumexp_async<fp32_fast_tf32_t>(int starpu_worker_hint, const Tile<fp32_fast_tf32_t> &src,
                                       const Tile<fp32_fast_tf32_t> &dst);

template
void logsumexp_async<fp32_fast_fp16_t>(int starpu_worker_hint, const Tile<fp32_fast_fp16_t> &src,
                                 const Tile<fp32_fast_fp16_t> &dst);

template
void logsumexp_async<fp32_fast_bf16_t>(int starpu_worker_hint, const Tile<fp32_fast_bf16_t> &src,
                                 const Tile<fp32_fast_bf16_t> &dst);

template
void logsumexp_async<fp64_t>(int starpu_worker_hint, const Tile<fp64_t> &src, const Tile<fp64_t> &dst);

template
void logsumexp_async<bf16_t>(int starpu_worker_hint, const Tile<bf16_t> &src, const Tile<bf16_t> &dst);

template
void logsumexp_async<fp16_t>(int starpu_worker_hint, const Tile<fp16_t> &src, const Tile<fp16_t> &dst);

// Explicit instantiation
template
void logsumexp<fp32_t>(int starpu_worker_hint, const Tile<fp32_t> &src, const Tile<fp32_t> &dst);

template
void logsumexp<fp32_fast_tf32_t>(int starpu_worker_hint, const Tile<fp32_fast_tf32_t> &src,
                                 const Tile<fp32_fast_tf32_t> &dst);

template
void logsumexp<fp32_fast_fp16_t>(int starpu_worker_hint, const Tile<fp32_fast_fp16_t> &src,
                                 const Tile<fp32_fast_fp16_t> &dst);

template
void logsumexp<fp32_fast_bf16_t>(int starpu_worker_hint, const Tile<fp32_fast_bf16_t> &src,
                                 const Tile<fp32_fast_bf16_t> &dst);

template
void logsumexp<fp64_t>(int starpu_worker_hint, const Tile<fp64_t> &src, const Tile<fp64_t> &dst);

template
void logsumexp<bf16_t>(int starpu_worker_hint, const Tile<bf16_t> &src, const Tile<bf16_t> &dst);

template
void logsumexp<fp16_t>(int starpu_worker_hint, const Tile<fp16_t> &src, const Tile<fp16_t> &dst);

} // namespace nntile::core
