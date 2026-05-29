/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/norm.cc
 * Euclidean norm of all elements in a Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tile/norm.hh"
#include "nntile/starpu/norm.hh"
#include "nntile/starpu/config.hh"

namespace nntile::tile
{

template<typename T>
void norm_async(Scalar alpha, const Tile<T> &src, Scalar beta, const Tile<T> &dst)
{
    // Check dimensions
    if(dst.ndim != 0)
    {
        throw std::runtime_error("dst.ndim != 0");
    }
    if(src.nelems == 0)
    {
        throw std::runtime_error("src.nelems == 0");
    }
    int mpi_rank = starpu_mpi_world_rank();
    int dst_rank = dst.mpi_get_rank();
    src.mpi_transfer(dst_rank, mpi_rank);
    if(mpi_rank == dst_rank)
    {
        // Insert task
        starpu::norm.submit<std::tuple<T>>(src.nelems, alpha, src, beta, dst);
    }
}

template<typename T>
void norm(Scalar alpha, const Tile<T> &src, Scalar beta, const Tile<T> &dst)
{
    norm_async<T>(alpha, src, beta, dst);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void norm_async<fp32_t>(Scalar alpha, const Tile<fp32_t> &src, Scalar beta,
        const Tile<fp32_t> &dst);

template
void norm_async<fp32_fast_tf32_t>(Scalar alpha, const Tile<fp32_fast_tf32_t> &src, Scalar beta,
        const Tile<fp32_fast_tf32_t> &dst);

template
void norm_async<fp32_fast_fp16_t>(Scalar alpha, const Tile<fp32_fast_fp16_t> &src, Scalar beta,
        const Tile<fp32_fast_fp16_t> &dst);

template
void norm_async<fp32_fast_bf16_t>(Scalar alpha, const Tile<fp32_fast_bf16_t> &src, Scalar beta,
        const Tile<fp32_fast_bf16_t> &dst);

template
void norm_async<fp64_t>(Scalar alpha, const Tile<fp64_t> &src, Scalar beta,
        const Tile<fp64_t> &dst);

template
void norm_async<bf16_t>(Scalar alpha, const Tile<bf16_t> &src, Scalar beta,
        const Tile<bf16_t> &dst);

template
void norm_async<fp16_t>(Scalar alpha, const Tile<fp16_t> &src, Scalar beta,
        const Tile<fp16_t> &dst);

// Explicit instantiation
template
void norm<fp32_t>(Scalar alpha, const Tile<fp32_t> &src, Scalar beta,
        const Tile<fp32_t> &dst);

template
void norm<fp32_fast_tf32_t>(Scalar alpha, const Tile<fp32_fast_tf32_t> &src, Scalar beta,
        const Tile<fp32_fast_tf32_t> &dst);

template
void norm<fp32_fast_fp16_t>(Scalar alpha, const Tile<fp32_fast_fp16_t> &src, Scalar beta,
        const Tile<fp32_fast_fp16_t> &dst);

template
void norm<fp32_fast_bf16_t>(Scalar alpha, const Tile<fp32_fast_bf16_t> &src, Scalar beta,
        const Tile<fp32_fast_bf16_t> &dst);

template
void norm<fp64_t>(Scalar alpha, const Tile<fp64_t> &src, Scalar beta,
        const Tile<fp64_t> &dst);

template
void norm<bf16_t>(Scalar alpha, const Tile<bf16_t> &src, Scalar beta,
        const Tile<bf16_t> &dst);

template
void norm<fp16_t>(Scalar alpha, const Tile<fp16_t> &src, Scalar beta,
        const Tile<fp16_t> &dst);

} // namespace nntile::tile
