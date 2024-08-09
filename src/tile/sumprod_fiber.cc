/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/sumprod_fiber.cc
 * Sums over fibers into a slice of a product of two Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tile/sumprod_fiber.hh"
#include "nntile/starpu/sumprod_fiber.hh"

namespace nntile::tile
{

template<typename T>
void sumprod_fiber_async(Scalar alpha, const Tile<T> &src1, const Tile<T> &src2,
        Scalar beta, const Tile<T> &dst, Index axis)
{
    // Check shapes of src1 and src2
    if(src1.shape != src2.shape)
    {
        throw std::runtime_error("src1.shape != src2.shape");
    }
    // Check dimensions
    if(dst.ndim != 1)
    {
        throw std::runtime_error("dst.ndim != 1");
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
    // Check shapes of src1 and dst
    if(src1.shape[axis] != dst.shape[0])
    {
        throw std::runtime_error("src1.shape[axis] != dst.shape[0]");
    }
    // Get sizes
    Index m, n, k;
    m = src1.stride[axis];
    n = src1.matrix_shape[axis+1][1];
    k = src1.shape[axis];
    // Insert task
    starpu::sumprod_fiber::submit<T>(m, n, k, alpha, src1, src2, beta, dst);
}

//! Tile-wise scalar products along outer axes
template<typename T>
void sumprod_fiber(Scalar alpha, const Tile<T> &src1, const Tile<T> &src2, Scalar beta,
        const Tile<T> &dst, Index axis)
{
    sumprod_fiber_async<T>(alpha, src1, src2, beta, dst, axis);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void sumprod_fiber_async<fp32_t>(Scalar alpha, const Tile<fp32_t> &src1,
        const Tile<fp32_t> &src2, Scalar beta, const Tile<fp32_t> &dst,
        Index axis);

template
void sumprod_fiber_async<fp32_fast_tf32_t>(Scalar alpha, const Tile<fp32_fast_tf32_t> &src1,
        const Tile<fp32_fast_tf32_t> &src2, Scalar beta, const Tile<fp32_fast_tf32_t> &dst,
        Index axis);

template
void sumprod_fiber_async<fp64_t>(Scalar alpha, const Tile<fp64_t> &src1,
        const Tile<fp64_t> &src2, Scalar beta, const Tile<fp64_t> &dst,
        Index axis);

template
void sumprod_fiber_async<bf16_t>(Scalar alpha, const Tile<bf16_t> &src1,
        const Tile<bf16_t> &src2, Scalar beta, const Tile<bf16_t> &dst,
        Index axis);

// Explicit instantiation
template
void sumprod_fiber<fp32_t>(Scalar alpha, const Tile<fp32_t> &src1,
        const Tile<fp32_t> &src2, Scalar beta, const Tile<fp32_t> &dst,
        Index axis);

template
void sumprod_fiber<fp32_fast_tf32_t>(Scalar alpha, const Tile<fp32_fast_tf32_t> &src1,
        const Tile<fp32_fast_tf32_t> &src2, Scalar beta, const Tile<fp32_fast_tf32_t> &dst,
        Index axis);

template
void sumprod_fiber<fp64_t>(Scalar alpha, const Tile<fp64_t> &src1,
        const Tile<fp64_t> &src2, Scalar beta, const Tile<fp64_t> &dst,
        Index axis);

template
void sumprod_fiber<bf16_t>(Scalar alpha, const Tile<bf16_t> &src1,
        const Tile<bf16_t> &src2, Scalar beta, const Tile<bf16_t> &dst,
        Index axis);

} // namespace nntile::tile
