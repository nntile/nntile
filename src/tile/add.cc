/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/add.cc
 * Add operation for two Tile<T>'s
 *
 * @version 1.1.0
 * */

#include "nntile/tile/add.hh"
#include "nntile/starpu/add.hh"
#include "nntile/starpu/config.hh"

namespace nntile::tile
{

//! Tile-wise add operation
template<typename T>
void add_async(Scalar alpha, const Tile<T> &src1, Scalar beta,
        const Tile<T> &src2, const Tile<T> &dst)
{
    // Check dimensions
    if(dst.ndim != src1.ndim)
    {
        throw std::runtime_error("dst.ndim != src.ndim");
    }
    if(dst.ndim != src2.ndim)
    {
        throw std::runtime_error("dst.ndim != src.ndim");
    }
    // Check shapes of tiles
    for(Index i = 0; i < dst.ndim; ++i)
    {
        if(dst.shape[i] != src1.shape[i])
        {
            throw std::runtime_error("dst.shape[i] != src1.shape[i]");
        }
    }
    for(Index i = 0; i < dst.ndim; ++i)
    {
        if(src2.shape[i] != src1.shape[i])
        {
            throw std::runtime_error("src2.shape[i] != src1.shape[i]");
        }
    }
    // Do nothing if alpha is zero and beta is one
    if(alpha == 0.0 && beta == 1.0)
    {
        return;
    }
    int mpi_rank = starpu_mpi_world_rank();
    int dst_rank = dst.mpi_get_rank();
    src1.mpi_transfer(dst_rank, mpi_rank);
    src2.mpi_transfer(dst_rank, mpi_rank);
    if(mpi_rank == dst_rank)
    {
        // Insert corresponding task
        starpu::add.submit<std::tuple<T>>(
            src1.nelems, alpha, src1, beta, src2, dst);
    }
}

//! Tile-wise add operation
template<typename T>
void add(Scalar alpha, const Tile<T> &src1, Scalar beta, const Tile<T> &src2,
        const Tile<T> &dst)
{
    add_async<T>(alpha, src1, beta, src2, dst);
    starpu_task_wait_for_all();
}

// Explicit instantiation of template
template
void add_async<fp32_t>(Scalar alpha, const Tile<fp32_t> &src1, Scalar beta,
        const Tile<fp32_t> &src2, const Tile<fp32_t> &dst);

template
void add_async<bf16_t>(Scalar alpha, const Tile<bf16_t> &src1, Scalar beta,
        const Tile<bf16_t> &src2, const Tile<bf16_t> &dst);

template
void add_async<fp16_t>(Scalar alpha, const Tile<fp16_t> &src1, Scalar beta,
        const Tile<fp16_t> &src2, const Tile<fp16_t> &dst);

template
void add_async<fp32_fast_tf32_t>(Scalar alpha,
        const Tile<fp32_fast_tf32_t> &src1, Scalar beta,
        const Tile<fp32_fast_tf32_t> &src2, const Tile<fp32_fast_tf32_t> &dst);

template
void add_async<fp32_fast_fp16_t>(Scalar alpha, const Tile<fp32_fast_fp16_t> &src1,
        Scalar beta, const Tile<fp32_fast_fp16_t> &src2,
        const Tile<fp32_fast_fp16_t> &dst);

template
void add_async<fp32_fast_bf16_t>(Scalar alpha, const Tile<fp32_fast_bf16_t> &src1,
        Scalar beta, const Tile<fp32_fast_bf16_t> &src2,
        const Tile<fp32_fast_bf16_t> &dst);

template
void add_async<fp64_t>(Scalar alpha, const Tile<fp64_t> &src1, Scalar beta,
        const Tile<fp64_t> &src2, const Tile<fp64_t> &dst);

// Explicit instantiation of template
template
void add<fp32_t>(Scalar alpha, const Tile<fp32_t> &src1, Scalar beta,
        const Tile<fp32_t> &src2, const Tile<fp32_t> &dst);

template
void add<bf16_t>(Scalar alpha, const Tile<bf16_t> &src1, Scalar beta,
        const Tile<bf16_t> &src2, const Tile<bf16_t> &dst);

template
void add<fp16_t>(Scalar alpha, const Tile<fp16_t> &src1, Scalar beta,
        const Tile<fp16_t> &src2, const Tile<fp16_t> &dst);

template
void add<fp32_fast_tf32_t>(Scalar alpha, const Tile<fp32_fast_tf32_t> &src1,
        Scalar beta, const Tile<fp32_fast_tf32_t> &src2,
        const Tile<fp32_fast_tf32_t> &dst);

template
void add<fp32_fast_fp16_t>(Scalar alpha, const Tile<fp32_fast_fp16_t> &src1,
        Scalar beta, const Tile<fp32_fast_fp16_t> &src2,
        const Tile<fp32_fast_fp16_t> &dst);

template
void add<fp32_fast_bf16_t>(Scalar alpha, const Tile<fp32_fast_bf16_t> &src1,
        Scalar beta, const Tile<fp32_fast_bf16_t> &src2,
        const Tile<fp32_fast_bf16_t> &dst);

template
void add<fp64_t>(Scalar alpha, const Tile<fp64_t> &src1, Scalar beta,
        const Tile<fp64_t> &src2, const Tile<fp64_t> &dst);

} // namespace nntile::tile
