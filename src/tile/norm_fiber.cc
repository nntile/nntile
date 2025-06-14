/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/norm_fiber.cc
 * Euclidean norms over slices into a fiber of a Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tile/norm_fiber.hh"
#include "nntile/starpu/norm_fiber.hh"

namespace nntile::tile
{

template<typename T>
void norm_fiber_async(Scalar alpha, const Tile<T> &src1, Scalar beta,
        const Tile<T> &src2,
        const Tile<T> &dst, Index axis, Index batch_ndim, int redux)
{
    // Check dimensions
    if(dst.ndim != batch_ndim+1)
    {
        throw std::runtime_error("dst.ndim != batch_ndim+1");
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
    if(axis >= ndim-batch_ndim)
    {
        throw std::runtime_error("axis >= ndim-batch_ndim");
    }
    // Check shapes
    if(dst.shape[0] != src1.shape[axis])
    {
        throw std::runtime_error("dst.shape[0] != src1.shape[axis]");
    }
    for(Index i = 0; i < batch_ndim; ++i)
    {
        if(dst.shape[i+1] != src1.shape[src1.ndim-batch_ndim+i])
        {
            throw std::runtime_error("dst.shape[i+1] != "
                    "src1.shape[src1.ndim-batch_ndim+i]");
        }
    }
    // Get sizes
    Index m, n, k, batch;
    batch = dst.matrix_shape[1][1];
    m = src1.stride[axis];
    n = src1.matrix_shape[axis+1][1] / batch;
    k = src1.shape[axis];
    // Insert task
    starpu::norm_fiber.submit<std::tuple<T>>(m, n, k, batch, alpha, src1, beta, src2, dst);
}

template<typename T>
void norm_fiber(Scalar alpha, const Tile<T> &src1, Scalar beta, const Tile<T> &src2, const Tile<T> &dst,
        Index axis, Index batch_ndim, int redux)
{
    norm_fiber_async<T>(alpha, src1, beta, src2, dst, axis, batch_ndim, redux);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void norm_fiber_async<fp32_t>(Scalar alpha, const Tile<fp32_t> &src1, Scalar beta,
        const Tile<fp32_t> &src2,
        const Tile<fp32_t> &dst, Index axis, Index batch_ndim, int redux=0);

template
void norm_fiber_async<fp32_fast_tf32_t>(Scalar alpha, const Tile<fp32_fast_tf32_t> &src1, Scalar beta,
        const Tile<fp32_fast_tf32_t> &src2,
        const Tile<fp32_fast_tf32_t> &dst, Index axis, Index batch_ndim, int redux=0);

template
void norm_fiber_async<fp32_fast_fp16_t>(Scalar alpha, const Tile<fp32_fast_fp16_t> &src1, Scalar beta,
        const Tile<fp32_fast_fp16_t> &src2,
        const Tile<fp32_fast_fp16_t> &dst, Index axis, Index batch_ndim, int redux=0);

template
void norm_fiber_async<fp32_fast_bf16_t>(Scalar alpha, const Tile<fp32_fast_bf16_t> &src1, Scalar beta,
        const Tile<fp32_fast_bf16_t> &src2,
        const Tile<fp32_fast_bf16_t> &dst, Index axis, Index batch_ndim, int redux=0);

template
void norm_fiber_async<fp64_t>(Scalar alpha, const Tile<fp64_t> &src1, Scalar beta,
        const Tile<fp64_t> &src2,
        const Tile<fp64_t> &dst, Index axis, Index batch_ndim, int redux=0);

template
void norm_fiber_async<bf16_t>(Scalar alpha, const Tile<bf16_t> &src1, Scalar beta,
        const Tile<bf16_t> &src2,
        const Tile<bf16_t> &dst, Index axis, Index batch_ndim, int redux=0);

// Explicit instantiation
template
void norm_fiber<fp32_t>(Scalar alpha, const Tile<fp32_t> &src1, Scalar beta,
        const Tile<fp32_t> &src2,
        const Tile<fp32_t> &dst, Index axis, Index batch_ndim, int redux=0);

template
void norm_fiber<fp32_fast_tf32_t>(Scalar alpha, const Tile<fp32_fast_tf32_t> &src1, Scalar beta,
        const Tile<fp32_fast_tf32_t> &src2,
        const Tile<fp32_fast_tf32_t> &dst, Index axis, Index batch_ndim, int redux=0);

template
void norm_fiber<fp32_fast_fp16_t>(Scalar alpha, const Tile<fp32_fast_fp16_t> &src1, Scalar beta,
        const Tile<fp32_fast_fp16_t> &src2,
        const Tile<fp32_fast_fp16_t> &dst, Index axis, Index batch_ndim, int redux=0);

template
void norm_fiber<fp32_fast_bf16_t>(Scalar alpha, const Tile<fp32_fast_bf16_t> &src1, Scalar beta,
        const Tile<fp32_fast_bf16_t> &src2,
        const Tile<fp32_fast_bf16_t> &dst, Index axis, Index batch_ndim, int redux=0);

template
void norm_fiber<fp64_t>(Scalar alpha, const Tile<fp64_t> &src1, Scalar beta,
        const Tile<fp64_t> &src2,
        const Tile<fp64_t> &dst, Index axis, Index batch_ndim, int redux=0);

template
void norm_fiber<bf16_t>(Scalar alpha, const Tile<bf16_t> &src1, Scalar beta,
        const Tile<bf16_t> &src2,
        const Tile<bf16_t> &dst, Index axis, Index batch_ndim, int redux=0);

} // namespace nntile::tile
