/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/core/scale_fiber.cc
 * Tile wrappers for scaling of a tensor with a broadcasted fiber
 *
 * @version 1.1.0
 * */

#include "nntile/core/scale_fiber.hh"
#include "nntile/starpu/scale_fiber.hh"
#include "nntile/starpu/config.hh"
#include "nntile/core/clear.hh"

namespace nntile::core
{

template<typename T>
void scale_fiber_async(int starpu_worker_hint, Scalar alpha, const Tile<T> &src, const Tile<T> &dst,
        Index axis, Index batch_ndim)
//! Tile<T> scaling of a tensor with a broadcasted fiber
/*! Reshapes input tensor and fiber into 3-dimensional and 1-dimensional arrays
 * and performs the following operations:
 *      dst[i,l,j,b] = alpha*src[l,b]
 *
 * @param[in] alpha: Scalar factor for src
 * @param[in] src: Input fiber, that is reshaped into 1D array
 * @param[out] dst: Resulting tensor, that is reshaped into 3D array
 * */
{
    // Check dimensions
    if(src.ndim != batch_ndim+1)
    {
        throw std::runtime_error("src.ndim != batch_ndim+1");
    }
    // Check axis
    if(axis < 0)
    {
        throw std::runtime_error("axis < 0");
    }
    if(axis < batch_ndim)
    {
        throw std::runtime_error("axis < batch_ndim");
    }
    if(axis >= dst.ndim)
    {
        throw std::runtime_error("axis >= dst.ndim");
    }
    // Check shapes of tiles
    if(src.shape[batch_ndim] != dst.shape[axis])
    {
        throw std::runtime_error("src.shape[batch_ndim] != dst.shape[axis]");
    }
    for(Index i = 0; i < batch_ndim; ++i)
    {
        if(src.shape[i] != dst.shape[i])
        {
            throw std::runtime_error("src.shape[i] != dst.shape[i]");
        }
    }
    // Reduce to clear the tile if alpha is zero
    if(alpha == 0.0)
    {
        clear_async(starpu_worker_hint, dst);
        return;
    }
    // Reshape inputs for simplicity: src -> (k,batch), dst -> (m,k,n,batch)
    Index m, n, k, batch;
    batch = src.matrix_shape[batch_ndim][0];
    m = dst.matrix_shape[axis+1][1];
    n = dst.matrix_shape[axis][0] / batch;
    k = dst.shape[axis];
    int mpi_rank = starpu_mpi_world_rank();
    int dst_rank = dst.mpi_get_rank();
    src.mpi_transfer(dst_rank, mpi_rank);
    if(mpi_rank == dst_rank)
    {
        // Insert corresponding task
        starpu::scale_fiber.submit<std::tuple<T>>(starpu_worker_hint, m, n, k, batch, alpha, src,
                dst);
    }
}

template<typename T>
void scale_fiber(int starpu_worker_hint, Scalar alpha, const Tile<T> &src, const Tile<T> &dst,
        Index axis, Index batch_ndim)
//! Tile<T> scaling of a tensor with a broadcasted fiber
/*! Blocking version of scale_fiber_async<T>.
 * Reshapes input tensor and fiber into 3-dimensional and 1-dimensional arrays
 * and performs the following operations:
 *      dst[i,l,j,b] = alpha*src[l,b]
 *
 * @param[in] alpha: Scalar factor for src
 * @param[in] src: Input fiber, that is reshaped into 1D array
 * @param[out] dst: Resulting tensor, that is reshaped into 3D array
 * */
{
    scale_fiber_async<T>(starpu_worker_hint, alpha, src, dst, axis, batch_ndim);
    nntile::starpu_task_wait_for_all_unless_deferred();
}

// Explicit instantiation of template
template
void scale_fiber_async<fp32_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_t> &src,
        const Tile<fp32_t> &dst, Index axis, Index batch_ndim);

template
void scale_fiber_async<fp32_fast_tf32_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_fast_tf32_t> &src,
        const Tile<fp32_fast_tf32_t> &dst, Index axis, Index batch_ndim);

template
void scale_fiber_async<fp32_fast_fp16_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_fast_fp16_t> &src,
        const Tile<fp32_fast_fp16_t> &dst, Index axis, Index batch_ndim);

template
void scale_fiber_async<fp32_fast_bf16_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_fast_bf16_t> &src,
        const Tile<fp32_fast_bf16_t> &dst, Index axis, Index batch_ndim);

template
void scale_fiber_async<fp64_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp64_t> &src,
        const Tile<fp64_t> &dst, Index axis, Index batch_ndim);

template
void scale_fiber_async<bf16_t>(int starpu_worker_hint, Scalar alpha, const Tile<bf16_t> &src,
        const Tile<bf16_t> &dst, Index axis, Index batch_ndim);

template
void scale_fiber_async<fp16_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp16_t> &src,
        const Tile<fp16_t> &dst, Index axis, Index batch_ndim);

// Explicit instantiation of template
template
void scale_fiber<fp32_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_t> &src,
        const Tile<fp32_t> &dst, Index axis, Index batch_ndim);

template
void scale_fiber<fp32_fast_tf32_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_fast_tf32_t> &src,
        const Tile<fp32_fast_tf32_t> &dst, Index axis, Index batch_ndim);

template
void scale_fiber<fp32_fast_fp16_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_fast_fp16_t> &src,
        const Tile<fp32_fast_fp16_t> &dst, Index axis, Index batch_ndim);

template
void scale_fiber<fp32_fast_bf16_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_fast_bf16_t> &src,
        const Tile<fp32_fast_bf16_t> &dst, Index axis, Index batch_ndim);

template
void scale_fiber<fp64_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp64_t> &src,
        const Tile<fp64_t> &dst, Index axis, Index batch_ndim);

template
void scale_fiber<bf16_t>(int starpu_worker_hint, Scalar alpha, const Tile<bf16_t> &src,
        const Tile<bf16_t> &dst, Index axis, Index batch_ndim);

template
void scale_fiber<fp16_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp16_t> &src,
        const Tile<fp16_t> &dst, Index axis, Index batch_ndim);

} // namespace nntile::core
