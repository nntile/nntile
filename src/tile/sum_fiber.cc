/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/sum_fiber.cc
 * Sums over slices into a fiber of a Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tile/sum_fiber.hh"
#include "nntile/starpu/sum_fiber.hh"

namespace nntile::tile
{

template<typename T>
void sum_fiber_async(Scalar alpha, const Tile<T> &src, Scalar beta,
        const Tile<T> &dst, Index axis, Index batch_ndim)
{
    // Check dimensions
    if(dst.ndim != batch_ndim+1)
    {
        throw std::runtime_error("dst.ndim != batch_ndim+1");
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
    if(axis >= ndim-batch_ndim)
    {
        throw std::runtime_error("axis >= ndim-batch_ndim");
    }
    // Check shapes
    if(dst.shape[0] != src.shape[axis])
    {
        throw std::runtime_error("dst.shape[0] != src.shape[axis]");
    }
    for(Index i = 0; i < batch_ndim; ++i)
    {
        if(dst.shape[i+1] != src.shape[src.ndim-batch_ndim+i])
        {
            throw std::runtime_error("dst.shape[i+1] != "
                    "src.shape[src.ndim-batch_ndim+i]");
        }
    }
    // Get sizes
    Index m, n, k, batch;
    batch = dst.matrix_shape[1][1];
    m = src.stride[axis];
    n = src.matrix_shape[axis+1][1] / batch;
    k = src.shape[axis];
    // Insert task
    starpu::sum_fiber::submit<T>(m, n, k, batch, alpha, src, beta, dst);
}

template<typename T>
void sum_fiber(Scalar alpha, const Tile<T> &src, Scalar beta, const Tile<T> &dst,
        Index axis, Index batch_ndim)
{
    sum_fiber_async<T>(alpha, src, beta, dst, axis, batch_ndim);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void sum_fiber_async<fp32_t>(Scalar alpha, const Tile<fp32_t> &src,
        Scalar beta, const Tile<fp32_t> &dst, Index axis, Index batch_ndim);

template
void sum_fiber_async<fp32_fast_tf32_t>(Scalar alpha, const Tile<fp32_fast_tf32_t> &src,
        Scalar beta, const Tile<fp32_fast_tf32_t> &dst, Index axis, Index batch_ndim);

template
void sum_fiber_async<fp64_t>(Scalar alpha, const Tile<fp64_t> &src,
        Scalar beta, const Tile<fp64_t> &dst, Index axis, Index batch_ndim);

template
void sum_fiber_async<bf16_t>(Scalar alpha, const Tile<bf16_t> &src, Scalar beta,
        const Tile<bf16_t> &dst, Index axis, Index batch_ndim);

// Explicit instantiation
template
void sum_fiber<fp32_t>(Scalar alpha, const Tile<fp32_t> &src, Scalar beta,
        const Tile<fp32_t> &dst, Index axis, Index batch_ndim);

template
void sum_fiber<fp32_fast_tf32_t>(Scalar alpha, const Tile<fp32_fast_tf32_t> &src, Scalar beta,
        const Tile<fp32_fast_tf32_t> &dst, Index axis, Index batch_ndim);

template
void sum_fiber<fp64_t>(Scalar alpha, const Tile<fp64_t> &src, Scalar beta,
        const Tile<fp64_t> &dst, Index axis, Index batch_ndim);

template
void sum_fiber<bf16_t>(Scalar alpha, const Tile<bf16_t> &src, Scalar beta,
        const Tile<bf16_t> &dst, Index axis, Index batch_ndim);

} // namespace nntile::tile
