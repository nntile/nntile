/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/core/multiply_slice.cc
 * Tile wrappers for per-element product of a tensor and a broadcasted slice
 *
 * @version 1.1.0
 * */

#include "nntile/core/multiply_slice.hh"
#ifdef NNTILE_USE_NNHAUL
#include "nntile/nnhaul/ops/multiply_slice.hh"
#else
#include "nntile/starpu/multiply_slice.hh"
#endif
#ifdef NNTILE_USE_NNHAUL
#include "nntile/nnhaul/sync_defer.hh"
#else
#include "nntile/starpu/config.hh"
#endif

namespace nntile::core
{

template<typename T>
void multiply_slice_async(int starpu_worker_hint, Scalar alpha, const Tile<T> &src, const Tile<T> &dst,
        Index axis)
//! Tile<T> per-element multiplication of a tensor and a broadcasted slice
/*! Reshapes input tensor and slice into 3-dimensional and 2-dimensional arrays
 * and performs the following operations:
 *      dst[i,l,j] = alpha * dst[i,l,j] * src[i,j]
 *
 * @param[in] alpha: Scalar factor
 * @param[in] src: Input slice, that is reshaped into 2D array
 * @param[inout] dst: Resulting tensor, that is reshaped into 3D array
 * */
{
    // Check dimensions
    if(dst.ndim != src.ndim+1)
    {
        throw std::runtime_error("dst.ndim != src.ndim+1");
    }
    // Check axis
    if(axis < 0)
    {
        throw std::runtime_error("axis < 0");
    }
    if(axis >= dst.ndim)
    {
        throw std::runtime_error("axis >= dst.ndim");
    }
    // Check shapes of tiles
    for(Index i = 0; i < axis; ++i)
    {
        if(dst.shape[i] != src.shape[i])
        {
            throw std::runtime_error("dst.shape[i] != src.shape[i]");
        }
    }
    for(Index i = axis+1; i < dst.ndim; ++i)
    {
        if(dst.shape[i] != src.shape[i-1])
        {
            throw std::runtime_error("dst.shape[i] != src.shape[i-1]");
        }
    }
    // Reshape inputs for simplicity: src -> (m,n), dst -> (m,k,n)
    Index m, n, k;
    m = dst.matrix_shape[axis+1][1];
    n = dst.matrix_shape[axis][0];
    k = dst.shape[axis];
    #ifdef NNTILE_USE_NNHAUL
    int mpi_rank = 0;
    #else
    int mpi_rank = starpu_mpi_world_rank();
    #endif
    #ifdef NNTILE_USE_NNHAUL
    int dst_rank = 0;
    #else
    int dst_rank = dst.mpi_get_rank();
    #endif
    #ifndef NNTILE_USE_NNHAUL
    src.mpi_transfer(dst_rank, mpi_rank);
    #endif
    if(mpi_rank == dst_rank)
    {
        // Insert corresponding task
        #ifdef NNTILE_USE_NNHAUL
        haul::multiply_slice.submit<std::tuple<T>>(starpu_worker_hint, m, n, k, alpha, src, dst);
        #else
        starpu::multiply_slice.submit<std::tuple<T>>(starpu_worker_hint, m, n, k, alpha, src, dst);
        #endif

    }
}

template<typename T>
void multiply_slice(int starpu_worker_hint, Scalar alpha, const Tile<T> &src, const Tile<T> &dst, Index axis)
//! Tile<T> per-element multiplication of a tensor and a broadcasted slice
/*! Blocking version of multiply_slice_async<T>.
 * Reshapes input tensor and slice into 3-dimensional and 2-dimensional arrays
 * and performs the following operations:
 *      dst[i,l,j] = alpha * dst[i,l,j] * src[i,j]
 *
 * @param[in] alpha: Scalar factor
 * @param[in] src: Input slice, that is reshaped into 2D array
 * @param[inout] dst: Resulting tensor, that is reshaped into 3D array
 * */
{
    multiply_slice_async<T>(starpu_worker_hint, alpha, src, dst, axis);
    #ifdef NNTILE_USE_NNHAUL
    nntile::nnhaul_task_wait_for_all_unless_deferred();
#else
    nntile::starpu_task_wait_for_all_unless_deferred();
#endif
}

// Explicit instantiation of template
template
void multiply_slice_async<fp32_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_t> &src,
        const Tile<fp32_t> &dst, Index axis);

template
void multiply_slice_async<fp32_fast_tf32_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_fast_tf32_t> &src,
        const Tile<fp32_fast_tf32_t> &dst, Index axis);

template
void multiply_slice_async<fp32_fast_fp16_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_fast_fp16_t> &src,
        const Tile<fp32_fast_fp16_t> &dst, Index axis);

template
void multiply_slice_async<fp32_fast_bf16_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_fast_bf16_t> &src,
        const Tile<fp32_fast_bf16_t> &dst, Index axis);

template
void multiply_slice_async<fp64_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp64_t> &src,
        const Tile<fp64_t> &dst, Index axis);

template
void multiply_slice_async<bf16_t>(int starpu_worker_hint, Scalar alpha, const Tile<bf16_t> &src,
        const Tile<bf16_t> &dst, Index axis);

template
void multiply_slice_async<fp16_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp16_t> &src,
        const Tile<fp16_t> &dst, Index axis);

// Explicit instantiation of template
template
void multiply_slice<fp32_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_t> &src,
        const Tile<fp32_t> &dst, Index axis);

template
void multiply_slice<fp32_fast_tf32_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_fast_tf32_t> &src,
        const Tile<fp32_fast_tf32_t> &dst, Index axis);

template
void multiply_slice<fp32_fast_fp16_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_fast_fp16_t> &src,
        const Tile<fp32_fast_fp16_t> &dst, Index axis);

template
void multiply_slice<fp32_fast_bf16_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_fast_bf16_t> &src,
        const Tile<fp32_fast_bf16_t> &dst, Index axis);

template
void multiply_slice<fp64_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp64_t> &src,
        const Tile<fp64_t> &dst, Index axis);

template
void multiply_slice<bf16_t>(int starpu_worker_hint, Scalar alpha, const Tile<bf16_t> &src,
        const Tile<bf16_t> &dst, Index axis);

template
void multiply_slice<fp16_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp16_t> &src,
        const Tile<fp16_t> &dst, Index axis);

} // namespace nntile::core
