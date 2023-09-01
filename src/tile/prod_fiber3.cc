/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/prod_fiber3.cc
 * Tile wrappers for per-element product of a tensor and a broadcasted fiber
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-07-03
 * */

#include "nntile/tile/prod_fiber3.hh"
#include "nntile/starpu/prod_fiber3.hh"

namespace nntile
{
namespace tile
{

template<typename T>
void prod_fiber3_async(const Tile<T> &src1, T alpha, const Tile<T> &src2,
        const Tile<T> &dst, Index axis)
//! Tile<T> per-element multiplication of a tensor and a broadcasted fiber
/*! Reshapes input tensor and fiber into 3-dimensional and 1-dimensional arrays
 * and performs the following operations:
 *      dst[i,l,j] = alpha * src1[l] * src2[i,l,j]
 *
 * @param[in] src: Input fiber, that is reshaped into 1D array
 * @param[in] alpha: Scalar factor
 * @param[inout] dst: Resulting tensor, that is reshaped into 3D array
 * */
{
    // Check dimensions
    if(src1.ndim != 1)
    {
        throw std::runtime_error("src1.ndim != 1");
    }
    if(src2.ndim != dst.ndim)
    {
        throw std::runtime_error("src2.ndim != dst.ndim");
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
    if(src1.shape[0] != dst.shape[axis])
    {
        throw std::runtime_error("src1.shape[0] != dst.shape[axis]");
    }
    if(src2.shape != dst.shape)
    {
        throw std::runtime_error("src2.shape != dst.shape");
    }
    // Reshape inputs for simplicity: src -> (m,n), dst -> (m,k,n)
    Index m, n, k;
    m = dst.stride[axis];
    n = dst.matrix_shape[axis+1][1];
    k = dst.shape[axis];
    // Insert corresponding task
    starpu::prod_fiber3::submit<T>(m, n, k, alpha, src1, src2, dst);
}

template<typename T>
void prod_fiber3(const Tile<T> &src1, T alpha, const Tile<T> &src2,
        const Tile<T> &dst, Index axis)
//! Tile<T> per-element multiplication of a tensor and a broadcasted fiber
/*! Blocking version of prod_fiber3_async<T>.
 * Reshapes input tensor and fiber into 3-dimensional and 1-dimensional arrays
 * and performs the following operations:
 *      dst[i,l,j] = alpha * src1[l] * src2[i,l,j]
 *
 * @param[in] src: Input fiber, that is reshaped into 1D array
 * @param[in] alpha: Scalar factor
 * @param[inout] dst: Resulting tensor, that is reshaped into 3D array
 * */
{
    prod_fiber3_async<T>(src1, alpha, src2, dst, axis);
    starpu_task_wait_for_all();
}

// Explicit instantiation of template
template
void prod_fiber3_async<fp32_t>(const Tile<fp32_t> &src1, fp32_t alpha,
        const Tile<fp32_t> &src2, const Tile<fp32_t> &dst, Index axis);

template
void prod_fiber3_async<fp64_t>(const Tile<fp64_t> &src1, fp64_t alpha,
        const Tile<fp64_t> &src2, const Tile<fp64_t> &dst, Index axis);

// Explicit instantiation of template
template
void prod_fiber3<fp32_t>(const Tile<fp32_t> &src1, fp32_t alpha,
        const Tile<fp32_t> &src2, const Tile<fp32_t> &dst, Index axis);

template
void prod_fiber3<fp64_t>(const Tile<fp64_t> &src1, fp64_t alpha,
        const Tile<fp64_t> &src2, const Tile<fp64_t> &dst, Index axis);

} // namespace tile
} // namespace nntile

