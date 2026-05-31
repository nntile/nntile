/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/core/silu.cc
 * SiLU operation for Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/core/silu.hh"
#include "nntile/starpu/silu.hh"
#include "nntile/starpu/config.hh"

namespace nntile::core
{

template<typename T>
void silu_async(int starpu_worker_hint, const Tile<T> &src, const Tile<T> &dst)
{
    // Check shapes
    if(src.shape != dst.shape)
    {
        throw std::runtime_error("src.shape != dst.shape");
    }
    int mpi_rank = starpu_mpi_world_rank();
    int dst_rank = dst.mpi_get_rank();
    src.mpi_transfer(dst_rank, mpi_rank);
    if(mpi_rank == dst_rank)
    {
        // Submit silu
        starpu::silu.submit<std::tuple<T>>(starpu_worker_hint, src.nelems, src, dst);
    }
}

template<typename T>
void silu(int starpu_worker_hint, const Tile<T> &src, const Tile<T> &dst)
{
    silu_async<T>(starpu_worker_hint, src, dst);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void silu_async<fp32_t>(int starpu_worker_hint, const Tile<fp32_t> &src,
        const Tile<fp32_t> &dst);

template
void silu_async<fp32_fast_tf32_t>(int starpu_worker_hint, const Tile<fp32_fast_tf32_t> &src,
        const Tile<fp32_fast_tf32_t> &dst);

template
void silu_async<fp32_fast_fp16_t>(int starpu_worker_hint, const Tile<fp32_fast_fp16_t> &src,
        const Tile<fp32_fast_fp16_t> &dst);

template
void silu_async<fp32_fast_bf16_t>(int starpu_worker_hint, const Tile<fp32_fast_bf16_t> &src,
        const Tile<fp32_fast_bf16_t> &dst);

template
void silu_async<fp64_t>(int starpu_worker_hint, const Tile<fp64_t> &src,
        const Tile<fp64_t> &dst);

template
void silu_async<bf16_t>(int starpu_worker_hint, const Tile<bf16_t> &src,
        const Tile<bf16_t> &dst);

template
void silu_async<fp16_t>(int starpu_worker_hint, const Tile<fp16_t> &src,
        const Tile<fp16_t> &dst);

// Explicit instantiation
template
void silu<fp32_t>(int starpu_worker_hint, const Tile<fp32_t> &src, const Tile<fp32_t> &dst);

template
void silu<fp32_fast_tf32_t>(int starpu_worker_hint, const Tile<fp32_fast_tf32_t> &src,
        const Tile<fp32_fast_tf32_t> &dst);

template
void silu<fp32_fast_fp16_t>(int starpu_worker_hint, const Tile<fp32_fast_fp16_t> &src,
        const Tile<fp32_fast_fp16_t> &dst);

template
void silu<fp32_fast_bf16_t>(int starpu_worker_hint, const Tile<fp32_fast_bf16_t> &src,
        const Tile<fp32_fast_bf16_t> &dst);

template
void silu<fp64_t>(int starpu_worker_hint, const Tile<fp64_t> &src, const Tile<fp64_t> &dst);

template
void silu<bf16_t>(int starpu_worker_hint, const Tile<bf16_t> &src, const Tile<bf16_t> &dst);

template
void silu<fp16_t>(int starpu_worker_hint, const Tile<fp16_t> &src, const Tile<fp16_t> &dst);

} // namespace nntile::core
