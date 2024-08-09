/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/add_slice.cc
 * Tile wrappers for addition of a tensor and a broadcasted slice
 *
 * @version 1.1.0
 * */

#include "nntile/tile/add_slice.hh"
#include "nntile/starpu/add_slice.hh"

namespace nntile::tile
{

template<typename T>
void add_slice_async(Scalar alpha, const Tile<T> &src, Scalar beta, const Tile<T> &dst,
        Index axis)
//! Tile<T> addition of a tensor and a broadcasted slice
/*! Reshapes input tensor and slice into 3-dimensional and 2-dimensional arrays
 * and performs the following operations:
 *      dst[i,l,j] = beta*dst[i,l,j] + alpha*src[i,j]
 *
 * @param[in] alpha: Scalar factor for src
 * @param[in] src: Input slice, that is reshaped into 2D array
 * @param[in] beta: Scaling factor for dst
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
    m = dst.stride[axis];
    n = dst.matrix_shape[axis+1][1];
    k = dst.shape[axis];
    // Insert corresponding task
    starpu::add_slice::submit<T>(m, n, k, alpha, src, beta, dst);
}

template<typename T>
void add_slice(Scalar alpha, const Tile<T> &src, Scalar beta, const Tile<T> &dst,
               Index axis)
//! Tile<T> addition of a tensor and a broadcasted slice
/*! Blocking version of add_slice_async<T>.
 * Reshapes input tensor and slice into 3-dimensional and 2-dimensional arrays
 * and performs the following operations:
 *      dst[i,l,j] = beta*dst[i,l,j] + alpha*src[i,j]
 *
 * @param[in] alpha: Scalar factor for src
 * @param[in] src: Input slice, that is reshaped into 2D array
 * @param[in] beta: Scaling factor for dst
 * @param[inout] dst: Resulting tensor, that is reshaped into 3D array
 * */
{
    add_slice_async<T>(alpha, src, beta, dst, axis);
    starpu_task_wait_for_all();
}

// Explicit instantiation of template
template
void add_slice_async<fp32_t>(Scalar alpha, const Tile<fp32_t> &src,
        Scalar beta, const Tile<fp32_t> &dst, Index axis);

template
void add_slice_async<fp64_t>(Scalar alpha, const Tile<fp64_t> &src,
        Scalar beta, const Tile<fp64_t> &dst, Index axis);

template
void add_slice_async<fp32_fast_tf32_t>(Scalar alpha, const Tile<fp32_fast_tf32_t> &src,
        Scalar beta, const Tile<fp32_fast_tf32_t> &dst, Index axis);

template
void add_slice_async<bf16_t>(Scalar alpha, const Tile<bf16_t> &src, Scalar beta,
        const Tile<bf16_t> &dst, Index axis);

// Explicit instantiation of template
template
void add_slice<fp32_t>(Scalar alpha, const Tile<fp32_t> &src, Scalar beta,
        const Tile<fp32_t> &dst, Index axis);

template
void add_slice<fp32_fast_tf32_t>(Scalar alpha, const Tile<fp32_fast_tf32_t> &src, Scalar beta,
        const Tile<fp32_fast_tf32_t> &dst, Index axis);

template
void add_slice<fp64_t>(Scalar alpha, const Tile<fp64_t> &src, Scalar beta,
        const Tile<fp64_t> &dst, Index axis);

template
void add_slice<bf16_t>(Scalar alpha, const Tile<bf16_t> &src, Scalar beta,
        const Tile<bf16_t> &dst, Index axis);

} // namespace nntile::tile
