/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/scale_slice.cc
 * Tile wrappers for scaling of a broadcasted slice
 *
 * @version 1.1.0
 * */

#include "nntile/tile/scale_slice.hh"
#include "nntile/starpu/scale_slice.hh"
#include "nntile/starpu/config.hh"

namespace nntile::tile
{

//! Asynchronous tile scaling of a broadcasted slice
template<typename T>
void scale_slice_async(Scalar alpha, const Tile<T> &src, const Tile<T> &dst, Index axis)
//! Tile<T> scaling of a broadcasted slice
/*! Reshapes input slice and dst tensor into 2-dimensional and 3-dimensional arrays
 * and performs the following operations:
 *      dst[i,l,j] = alpha*src[i,j]
 *
 * @param[in] alpha: Scalar factor for src
 * @param[in] src: Input slice, that is reshaped into 2D array
 * @param[out] dst: Resulting tensor, that is reshaped into 3D array
 * @param[in] axis: Axis along which the slice is broadcasted
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
    m = dst.stride[axis];
    n = dst.matrix_shape[axis+1][1];
    k = dst.shape[axis];
    int mpi_rank = starpu_mpi_world_rank();
    int dst_rank = dst.mpi_get_rank();
    src.mpi_transfer(dst_rank, mpi_rank);
    if(mpi_rank == dst_rank)
    {
        // Insert corresponding task
        starpu::scale_slice.submit<std::tuple<T>>(m, n, k, alpha, src, dst);
    }
}

//! Blocking version of tile scaling of a broadcasted slice
template<typename T>
void scale_slice(Scalar alpha, const Tile<T> &src, const Tile<T> &dst, Index axis)
//! Tile<T> scaling of a broadcasted slice
/*! Blocking version of scale_slice_async<T>.
 * Reshapes input slice and dst tensor into 2-dimensional and 3-dimensional arrays
 * and performs the following operations:
 *      dst[i,l,j] = alpha*src[i,j]
 *
 * @param[in] alpha: Scalar factor for src
 * @param[in] src: Input slice, that is reshaped into 2D array
 * @param[out] dst: Resulting tensor, that is reshaped into 3D array
 * @param[in] axis: Axis along which the slice is broadcasted
 * */
{
    scale_slice_async<T>(alpha, src, dst, axis);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void scale_slice_async<fp64_t>(Scalar alpha, const Tile<fp64_t> &src, const Tile<fp64_t> &dst, Index axis);

template
void scale_slice_async<fp32_t>(Scalar alpha, const Tile<fp32_t> &src, const Tile<fp32_t> &dst, Index axis);

template
void scale_slice_async<fp32_fast_tf32_t>(Scalar alpha, const Tile<fp32_fast_tf32_t> &src, const Tile<fp32_fast_tf32_t> &dst, Index axis);

template
void scale_slice_async<fp32_fast_fp16_t>(Scalar alpha, const Tile<fp32_fast_fp16_t> &src, const Tile<fp32_fast_fp16_t> &dst, Index axis);

template
void scale_slice_async<fp32_fast_bf16_t>(Scalar alpha, const Tile<fp32_fast_bf16_t> &src, const Tile<fp32_fast_bf16_t> &dst, Index axis);

template
void scale_slice_async<bf16_t>(Scalar alpha, const Tile<bf16_t> &src, const Tile<bf16_t> &dst, Index axis);

template
void scale_slice_async<fp16_t>(Scalar alpha, const Tile<fp16_t> &src, const Tile<fp16_t> &dst, Index axis);

template
void scale_slice<fp64_t>(Scalar alpha, const Tile<fp64_t> &src, const Tile<fp64_t> &dst, Index axis);

template
void scale_slice<fp32_t>(Scalar alpha, const Tile<fp32_t> &src, const Tile<fp32_t> &dst, Index axis);

template
void scale_slice<fp32_fast_tf32_t>(Scalar alpha, const Tile<fp32_fast_tf32_t> &src, const Tile<fp32_fast_tf32_t> &dst, Index axis);

template
void scale_slice<fp32_fast_fp16_t>(Scalar alpha, const Tile<fp32_fast_fp16_t> &src, const Tile<fp32_fast_fp16_t> &dst, Index axis);

template
void scale_slice<fp32_fast_bf16_t>(Scalar alpha, const Tile<fp32_fast_bf16_t> &src, const Tile<fp32_fast_bf16_t> &dst, Index axis);

template
void scale_slice<bf16_t>(Scalar alpha, const Tile<bf16_t> &src, const Tile<bf16_t> &dst, Index axis);

template
void scale_slice<fp16_t>(Scalar alpha, const Tile<fp16_t> &src, const Tile<fp16_t> &dst, Index axis);

} // namespace nntile::tile
