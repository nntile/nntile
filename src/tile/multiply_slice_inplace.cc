/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/multiply_slice_inplace.cc
 * Tile wrappers for in-place multiplication of a tensor and a broadcasted slice
 *
 * @version 1.1.0
 * */

#include "nntile/tile/multiply_slice_inplace.hh"
#include "nntile/starpu/multiply_slice_inplace.hh"
#include "nntile/starpu/config.hh"

namespace nntile::tile
{

template<typename T>
void multiply_slice_inplace_async(Scalar alpha, const Tile<T> &src, Scalar beta,
        const Tile<T> &dst, Index axis)
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
    if(dst.shape[axis] != src.shape[axis])
    {
        throw std::runtime_error("dst.shape[axis] != src.shape[axis]");
    }
    for(Index i = axis+1; i < dst.ndim; ++i)
    {
        if(dst.shape[i] != src.shape[i-1])
        {
            throw std::runtime_error("dst.shape[i] != src.shape[i-1]");
        }
    }
    // Submit task
    multiply_slice_inplace.submit<std::tuple<T>>(dst.shape[0], dst.shape[1], dst.shape[2],
            alpha, src, beta, dst, axis);
}

template<typename T>
void multiply_slice_inplace(Scalar alpha, const Tile<T> &src, Scalar beta, const Tile<T> &dst,
        Index axis)
{
    multiply_slice_inplace_async<T>(alpha, src, beta, dst, axis);
    starpu_task_wait_for_all();
}

// Explicit instantiation for all supported types
template void multiply_slice_inplace<fp32_t>(Scalar alpha, const Tile<fp32_t> &src, Scalar beta, const Tile<fp32_t> &dst, Index axis);
template void multiply_slice_inplace<fp64_t>(Scalar alpha, const Tile<fp64_t> &src, Scalar beta, const Tile<fp64_t> &dst, Index axis);
template void multiply_slice_inplace<bf16_t>(Scalar alpha, const Tile<bf16_t> &src, Scalar beta, const Tile<bf16_t> &dst, Index axis);
template void multiply_slice_inplace<fp16_t>(Scalar alpha, const Tile<fp16_t> &src, Scalar beta, const Tile<fp16_t> &dst, Index axis);

} // namespace nntile::tile