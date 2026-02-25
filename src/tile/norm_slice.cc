/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/norm_slice.cc
 * Euclidean norms of fibers into a slice of a Tile<T> (out-of-place version)
 *
 * @version 1.1.0
 * */

#include "nntile/tile/norm_slice.hh"
#include "nntile/starpu/norm_slice.hh"
#include "nntile/starpu/config.hh"

namespace nntile::tile
{

template<typename T>
void norm_slice_async(Scalar alpha, const Tile<T> &src1, Scalar beta, const Tile<T> &src2,
        const Tile<T> &dst, Index axis, int redux)
{
    // Check dimensions
    if(src1.ndim-1 != src2.ndim)
    {
        throw std::runtime_error("src1.ndim-1 != src2.ndim");
    }
    if(src2.ndim != dst.ndim)
    {
        throw std::runtime_error("src2.ndim != dst.ndim");
    }
    Index ndim = src1.ndim;
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
    // Check shapes of src1, src2 and dst
    for(Index i = 0; i < axis; i++)
    {
        if(src1.shape[i] != src2.shape[i])
        {
            throw std::runtime_error("src1.shape[i] != src2.shape[i]");
        }
        if(src1.shape[i] != dst.shape[i])
        {
            throw std::runtime_error("src1.shape[i] != dst.shape[i]");
        }
    }
    for(Index i = axis+1; i < ndim; i++)
    {
        if(src1.shape[i] != src2.shape[i-1])
        {
            throw std::runtime_error("src1.shape[i] != src2.shape[i-1]");
        }
        if(src1.shape[i] != dst.shape[i-1])
        {
            throw std::runtime_error("src1.shape[i] != dst.shape[i-1]");
        }
    }
    // Get sizes
    Index m, n, k;
    m = src1.stride[axis];
    n = src1.matrix_shape[axis+1][1];
    k = src1.shape[axis];
    // Insert task
    int mpi_rank = starpu_mpi_world_rank();
    int dst_rank = dst.mpi_get_rank();
    src1.mpi_transfer(dst_rank, mpi_rank);
    src2.mpi_transfer(dst_rank, mpi_rank);
    if(mpi_rank == dst_rank)
    {
        starpu::norm_slice.submit<std::tuple<T>>(m, n, k, alpha, src1, beta,
                src2, dst, redux);
    }
}

template<typename T>
void norm_slice(Scalar alpha, const Tile<T> &src1, Scalar beta, const Tile<T> &src2,
        const Tile<T> &dst, Index axis, int redux)
{
    norm_slice_async<T>(alpha, src1, beta, src2, dst, axis, redux);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void norm_slice_async<fp32_t>(Scalar alpha, const Tile<fp32_t> &src1, Scalar beta,
        const Tile<fp32_t> &src2, const Tile<fp32_t> &dst, Index axis,
        int redux);

template
void norm_slice_async<fp32_fast_tf32_t>(Scalar alpha, const Tile<fp32_fast_tf32_t> &src1, Scalar beta,
        const Tile<fp32_fast_tf32_t> &src2, const Tile<fp32_fast_tf32_t> &dst,
        Index axis, int redux);

template
void norm_slice_async<fp32_fast_fp16_t>(Scalar alpha, const Tile<fp32_fast_fp16_t> &src1, Scalar beta,
        const Tile<fp32_fast_fp16_t> &src2, const Tile<fp32_fast_fp16_t> &dst,
        Index axis, int redux);

template
void norm_slice_async<fp32_fast_bf16_t>(Scalar alpha, const Tile<fp32_fast_bf16_t> &src1, Scalar beta,
        const Tile<fp32_fast_bf16_t> &src2, const Tile<fp32_fast_bf16_t> &dst,
        Index axis, int redux);

template
void norm_slice_async<fp64_t>(Scalar alpha, const Tile<fp64_t> &src1, Scalar beta,
        const Tile<fp64_t> &src2, const Tile<fp64_t> &dst, Index axis,
        int redux);

template
void norm_slice_async<bf16_t>(Scalar alpha, const Tile<bf16_t> &src1, Scalar beta,
        const Tile<bf16_t> &src2, const Tile<bf16_t> &dst, Index axis,
        int redux);

template
void norm_slice_async<fp16_t>(Scalar alpha, const Tile<fp16_t> &src1, Scalar beta,
        const Tile<fp16_t> &src2, const Tile<fp16_t> &dst, Index axis,
        int redux);

// Explicit instantiation
template
void norm_slice<fp32_t>(Scalar alpha, const Tile<fp32_t> &src1, Scalar beta,
        const Tile<fp32_t> &src2, const Tile<fp32_t> &dst, Index axis,
        int redux);

template
void norm_slice<fp32_fast_tf32_t>(Scalar alpha, const Tile<fp32_fast_tf32_t> &src1, Scalar beta,
        const Tile<fp32_fast_tf32_t> &src2, const Tile<fp32_fast_tf32_t> &dst,
        Index axis, int redux);

template
void norm_slice<fp32_fast_fp16_t>(Scalar alpha, const Tile<fp32_fast_fp16_t> &src1, Scalar beta,
        const Tile<fp32_fast_fp16_t> &src2, const Tile<fp32_fast_fp16_t> &dst,
        Index axis, int redux);

template
void norm_slice<fp32_fast_bf16_t>(Scalar alpha, const Tile<fp32_fast_bf16_t> &src1, Scalar beta,
        const Tile<fp32_fast_bf16_t> &src2, const Tile<fp32_fast_bf16_t> &dst,
        Index axis, int redux);

template
void norm_slice<fp64_t>(Scalar alpha, const Tile<fp64_t> &src1, Scalar beta,
        const Tile<fp64_t> &src2, const Tile<fp64_t> &dst, Index axis,
        int redux);

template
void norm_slice<bf16_t>(Scalar alpha, const Tile<bf16_t> &src1, Scalar beta,
        const Tile<bf16_t> &src2, const Tile<bf16_t> &dst, Index axis,
        int redux);

template
void norm_slice<fp16_t>(Scalar alpha, const Tile<fp16_t> &src1, Scalar beta,
        const Tile<fp16_t> &src2, const Tile<fp16_t> &dst, Index axis,
        int redux);

} // namespace nntile::tile
