/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/prod_fiber.cc
 * Tile wrappers for per-element product of a tensor and a broadcasted fiber
 *
 * @version 1.1.0
 * */

#include "nntile/tile/prod_fiber.hh"
#include "nntile/starpu/prod_fiber.hh"

namespace nntile::tile
{

template<typename T>
void prod_fiber_async(const Tile<T> &src, Scalar alpha, const Tile<T> &dst,
        Index axis)
//! Tile<T> per-element multiplication of a tensor and a broadcasted fiber
/*! Reshapes input tensor and fiber into 3-dimensional and 1-dimensional arrays
 * and performs the following operations:
 *      dst[i,l,j] = alpha * dst[i,l,j] * src[l]
 *
 * @param[in] src: Input fiber, that is reshaped into 1D array
 * @param[in] alpha: Scalar factor
 * @param[inout] dst: Resulting tensor, that is reshaped into 3D array
 * */
{
    // Check dimensions
    if(src.ndim != 1)
    {
        throw std::runtime_error("src.ndim != 1");
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
    if(src.shape[0] != dst.shape[axis])
    {
        throw std::runtime_error("src.shape[0] != dst.shape[axis]");
    }
    // Reshape inputs for simplicity: src -> (m,n), dst -> (m,k,n)
    Index m, n, k;
    m = dst.stride[axis];
    n = dst.matrix_shape[axis+1][1];
    k = dst.shape[axis];
    // Insert corresponding task
    starpu::prod_fiber::submit<T>(m, n, k, alpha, src, dst);
}

template<typename T>
void prod_fiber(const Tile<T> &src, Scalar alpha, const Tile<T> &dst, Index axis)
//! Tile<T> per-element multiplication of a tensor and a broadcasted fiber
/*! Blocking version of prod_fiber_async<T>.
 * Reshapes input tensor and fiber into 3-dimensional and 1-dimensional arrays
 * and performs the following operations:
 *      dst[i,l,j] = alpha * dst[i,l,j] * src[l]
 *
 * @param[in] src: Input fiber, that is reshaped into 1D array
 * @param[in] alpha: Scalar factor
 * @param[inout] dst: Resulting tensor, that is reshaped into 3D array
 * */
{
    prod_fiber_async<T>(src, alpha, dst, axis);
    starpu_task_wait_for_all();
}

// Explicit instantiation of template
template
void prod_fiber_async<fp32_t>(const Tile<fp32_t> &src, Scalar alpha,
        const Tile<fp32_t> &dst, Index axis);

template
void prod_fiber_async<fp64_t>(const Tile<fp64_t> &src, Scalar alpha,
        const Tile<fp64_t> &dst, Index axis);

// Explicit instantiation of template
template
void prod_fiber<fp32_t>(const Tile<fp32_t> &src, Scalar alpha,
        const Tile<fp32_t> &dst, Index axis);

template
void prod_fiber<fp64_t>(const Tile<fp64_t> &src, Scalar alpha,
        const Tile<fp64_t> &dst, Index axis);

} // namespace nntile::tile
