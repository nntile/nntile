/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/sum_slice.cc
 * Sum over fibers into a slice of a Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tile/sum_slice.hh"
#include "nntile/starpu/sum_slice.hh"
#include "nntile/starpu/config.hh"

namespace nntile::tile
{

//! Tile-wise sum_slice
template<typename T>
void sum_slice_async(Scalar alpha, const Tile<T> &src, Scalar beta,
        const Tile<T> &dst, Index axis, int redux)
{
    // Check dimensions
    if(src.ndim - 1 != dst.ndim) // before was src.ndim != dst.ndim
    {
        throw std::runtime_error("src.ndim -1 != dst.ndim");
    }
    Index ndim = src.ndim;
    // Treat special case of ndim=0
    if(ndim == 0)
    {
        throw std::runtime_error("Scalar input makes no sense");
    }
    // Check axis
    if(axis < 0)
    {
        throw std::runtime_error("axis < 0");
    }
    if(axis >= ndim)
    {
        throw std::runtime_error("axis >= ndim");
    }

    // check if axis consisted, using two pointers
    for(Index i = 0, j = 0; i < src.ndim; i++)
    {
        if (i == axis) {
            continue;
        }
        if(src.shape[i] != dst.shape[j])
        {
            throw std::runtime_error("src.shape[i] != dst.shape[j]");
        }
        j++;
    }
    // Get sizes
    Index m, n, k;
    m = src.stride[axis];
    n = src.matrix_shape[axis+1][1];
    k = src.shape[axis];
    // Insert task
    int mpi_rank = starpu_mpi_world_rank();
    int dst_rank = dst.mpi_get_rank();
    src.mpi_transfer(dst_rank, mpi_rank);
    if(mpi_rank == dst_rank)
    {
        starpu::sum_slice.submit<std::tuple<T>>(m, n, k, alpha, src, beta, dst,
                redux);
    }
}

//! Tile-wise sum_slice
template<typename T>
void sum_slice(Scalar alpha, const Tile<T> &src, Scalar beta, const Tile<T> &dst,
        Index axis, int redux)
{
    sum_slice_async<T>(alpha, src, beta, dst, axis, redux);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void sum_slice_async<fp32_t>(Scalar alpha, const Tile<fp32_t> &src,
        Scalar beta, const Tile<fp32_t> &dst, Index axis, int redux);

template
void sum_slice_async<bf16_t>(Scalar alpha, const Tile<bf16_t> &src, Scalar beta,
        const Tile<bf16_t> &dst, Index axis, int redux);

template
void sum_slice_async<fp16_t>(Scalar alpha, const Tile<fp16_t> &src, Scalar beta,
        const Tile<fp16_t> &dst, Index axis, int redux);

template
void sum_slice_async<fp32_fast_tf32_t>(Scalar alpha, const Tile<fp32_fast_tf32_t> &src,
        Scalar beta, const Tile<fp32_fast_tf32_t> &dst, Index axis, int redux);

template
void sum_slice_async<fp32_fast_fp16_t>(Scalar alpha, const Tile<fp32_fast_fp16_t> &src, Scalar beta,
        const Tile<fp32_fast_fp16_t> &dst, Index axis, int redux);

template
void sum_slice_async<fp32_fast_bf16_t>(Scalar alpha, const Tile<fp32_fast_bf16_t> &src, Scalar beta,
        const Tile<fp32_fast_bf16_t> &dst, Index axis, int redux);

template
void sum_slice_async<fp64_t>(Scalar alpha, const Tile<fp64_t> &src,
        Scalar beta, const Tile<fp64_t> &dst, Index axis, int redux);

// Explicit instantiation
template
void sum_slice<fp32_t>(Scalar alpha, const Tile<fp32_t> &src, Scalar beta,
        const Tile<fp32_t> &dst, Index axis, int redux);

template
void sum_slice<fp32_fast_tf32_t>(Scalar alpha, const Tile<fp32_fast_tf32_t> &src, Scalar beta,
        const Tile<fp32_fast_tf32_t> &dst, Index axis, int redux);

template
void sum_slice<fp32_fast_fp16_t>(Scalar alpha, const Tile<fp32_fast_fp16_t> &src, Scalar beta,
        const Tile<fp32_fast_fp16_t> &dst, Index axis, int redux);

template
void sum_slice<fp32_fast_bf16_t>(Scalar alpha, const Tile<fp32_fast_bf16_t> &src, Scalar beta,
        const Tile<fp32_fast_bf16_t> &dst, Index axis, int redux);

template
void sum_slice<fp64_t>(Scalar alpha, const Tile<fp64_t> &src, Scalar beta,
        const Tile<fp64_t> &dst, Index axis, int redux);

template
void sum_slice<bf16_t>(Scalar alpha, const Tile<bf16_t> &src, Scalar beta,
        const Tile<bf16_t> &dst, Index axis, int redux);

template
void sum_slice<fp16_t>(Scalar alpha, const Tile<fp16_t> &src, Scalar beta,
        const Tile<fp16_t> &dst, Index axis, int redux);

} // namespace nntile::tile
