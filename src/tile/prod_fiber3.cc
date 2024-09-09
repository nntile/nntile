/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/prod_fiber3.cc
 * Tile wrappers for per-element product of a tensor and a broadcasted fiber
 *
 * @version 1.1.0
 * */

#include "nntile/tile/prod_fiber3.hh"
#include "nntile/starpu/prod_fiber3.hh"

namespace nntile::tile
{

template<typename T>
void prod_fiber3_async(const Tile<T> &src1, Scalar alpha, const Tile<T> &src2,
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
void prod_fiber3(const Tile<T> &src1, Scalar alpha, const Tile<T> &src2,
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
void prod_fiber3_async<fp32_t>(const Tile<fp32_t> &src1, Scalar alpha,
        const Tile<fp32_t> &src2, const Tile<fp32_t> &dst, Index axis);

template
void prod_fiber3_async<fp32_fast_tf32_t>(const Tile<fp32_fast_tf32_t> &src1, Scalar alpha,
        const Tile<fp32_fast_tf32_t> &src2, const Tile<fp32_fast_tf32_t> &dst, Index axis);

template
void prod_fiber3_async<fp32_fast_fp16_t>(const Tile<fp32_fast_fp16_t> &src1, Scalar alpha,
        const Tile<fp32_fast_fp16_t> &src2, const Tile<fp32_fast_fp16_t> &dst, Index axis);

template
void prod_fiber3_async<fp32_fast_bf16_t>(const Tile<fp32_fast_bf16_t> &src1, Scalar alpha,
        const Tile<fp32_fast_bf16_t> &src2, const Tile<fp32_fast_bf16_t> &dst, Index axis);

template
void prod_fiber3_async<fp64_t>(const Tile<fp64_t> &src1, Scalar alpha,
        const Tile<fp64_t> &src2, const Tile<fp64_t> &dst, Index axis);

template
void prod_fiber3_async<bf16_t>(const Tile<bf16_t> &src1, Scalar alpha,
        const Tile<bf16_t> &src2, const Tile<bf16_t> &dst, Index axis);

// Explicit instantiation of template
template
void prod_fiber3<fp32_t>(const Tile<fp32_t> &src1, Scalar alpha,
        const Tile<fp32_t> &src2, const Tile<fp32_t> &dst, Index axis);

template
void prod_fiber3<fp32_fast_tf32_t>(const Tile<fp32_fast_tf32_t> &src1, Scalar alpha,
        const Tile<fp32_fast_tf32_t> &src2, const Tile<fp32_fast_tf32_t> &dst, Index axis);

template
void prod_fiber3<fp32_fast_fp16_t>(const Tile<fp32_fast_fp16_t> &src1, Scalar alpha,
        const Tile<fp32_fast_fp16_t> &src2, const Tile<fp32_fast_fp16_t> &dst, Index axis);

template
void prod_fiber3<fp32_fast_bf16_t>(const Tile<fp32_fast_bf16_t> &src1, Scalar alpha,
        const Tile<fp32_fast_bf16_t> &src2, const Tile<fp32_fast_bf16_t> &dst, Index axis);

template
void prod_fiber3<fp64_t>(const Tile<fp64_t> &src1, Scalar alpha,
        const Tile<fp64_t> &src2, const Tile<fp64_t> &dst, Index axis);

template
void prod_fiber3<bf16_t>(const Tile<bf16_t> &src1, Scalar alpha,
        const Tile<bf16_t> &src2, const Tile<bf16_t> &dst, Index axis);

} // namespace nntile::tile
