/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/hypot.cc
 * hypot operation for Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tile/hypot.hh"
#include "nntile/starpu/hypot.hh"

namespace nntile::tile
{

//! Tile-wise hypot operation
template<typename T>
void hypot_async(Scalar alpha, const Tile<T> &src, Scalar beta, const Tile<T> &dst)
{
    // Check dimensions
    if(dst.ndim != src.ndim)
    {
        throw std::runtime_error("dst.ndim != src.ndim");
    }
    // Check shapes of tiles
    for(Index i = 0; i < dst.ndim; ++i)
    {
        if(dst.shape[i] != src.shape[i])
        {
            throw std::runtime_error("dst.shape[i] != src.shape[i]");
        }
    }
    // Do nothing if alpha is zero and beta is one
    if(alpha == 0.0 && beta == 1.0)
    {
        return;
    }
    // Insert corresponding task
    starpu::hypot::submit<T>(src.nelems, alpha, src, beta, dst);
}

//! Tile-wise hypot operation
template<typename T>
void hypot(Scalar alpha, const Tile<T> &src, Scalar beta, const Tile<T> &dst)
{
    hypot_async<T>(alpha, src, beta, dst);
    starpu_task_wait_for_all();
}

// Explicit instantiation of template
template
void hypot_async<fp32_t>(Scalar alpha, const Tile<fp32_t> &src, Scalar beta,
        const Tile<fp32_t> &dst);

template
void hypot_async<fp32_fast_tf32_t>(Scalar alpha, const Tile<fp32_fast_tf32_t> &src, Scalar beta,
        const Tile<fp32_fast_tf32_t> &dst);

template
void hypot_async<fp64_t>(Scalar alpha, const Tile<fp64_t> &src, Scalar beta,
        const Tile<fp64_t> &dst);

template
void hypot_async<bf16_t>(Scalar alpha, const Tile<bf16_t> &src, Scalar beta,
        const Tile<bf16_t> &dst);

// Explicit instantiation of template
template
void hypot<fp32_t>(Scalar alpha, const Tile<fp32_t> &src, Scalar beta,
        const Tile<fp32_t> &dst);

template
void hypot<fp32_fast_tf32_t>(Scalar alpha, const Tile<fp32_fast_tf32_t> &src, Scalar beta,
        const Tile<fp32_fast_tf32_t> &dst);

template
void hypot<fp64_t>(Scalar alpha, const Tile<fp64_t> &src, Scalar beta,
        const Tile<fp64_t> &dst);

template
void hypot<bf16_t>(Scalar alpha, const Tile<bf16_t> &src, Scalar beta,
        const Tile<bf16_t> &dst);

} // namespace nntile::tile
