/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/logsumexp.cc
 * Max and sum of exponents of Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tile/logsumexp.hh"
#include "nntile/starpu/logsumexp.hh"

namespace nntile::tile
{

template<typename T>
void logsumexp_async(const Tile<T> &src, const Tile<T> &dst)
// TODO - add description
{
    // Check dimensions
    if(src.ndim - 1 != dst.ndim)
    {
        throw std::runtime_error("src.ndim - 1 != dst.ndim");
    }
    Index ndim = src.ndim;
    // Treat special case of ndim=0
    if(ndim == 0)
    {
        throw std::runtime_error("Scalar input makes no sense");
    }
    if(src.shape[0] != 2)
    {
        throw std::runtime_error("src.shape[0] != 2");
    }
    for(Index i = 0; i < ndim - 1; ++i)
    {
        if (src.shape[i+1] != dst.shape[i])
        {
            throw std::runtime_error("src.shape[i+1] != dst.shape[i]");
        }
    }
    // Insert task
    starpu::logsumexp::submit<T>(dst.nelems, src, dst);
}

//! Tile-wise logsumexp
template<typename T>
void logsumexp(const Tile<T> &src, const Tile<T> &dst)
{
    logsumexp_async<T>(src, dst);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void logsumexp_async<fp32_t>(const Tile<fp32_t> &src, const Tile<fp32_t> &dst);

template
void logsumexp_async<fp32_fast_tf32_t>(const Tile<fp32_fast_tf32_t> &src, const Tile<fp32_fast_tf32_t> &dst);


template
void logsumexp_async<fp64_t>(const Tile<fp64_t> &src, const Tile<fp64_t> &dst);

template
void logsumexp_async<bf16_t>(const Tile<bf16_t> &src, const Tile<bf16_t> &dst);

// Explicit instantiation
template
void logsumexp<fp32_t>(const Tile<fp32_t> &src, const Tile<fp32_t> &dst);

template
void logsumexp<fp32_fast_tf32_t>(const Tile<fp32_fast_tf32_t> &src, const Tile<fp32_fast_tf32_t> &dst);

template
void logsumexp<fp64_t>(const Tile<fp64_t> &src, const Tile<fp64_t> &dst);

template
void logsumexp<bf16_t>(const Tile<bf16_t> &src, const Tile<bf16_t> &dst);

} // namespace nntile::tile
