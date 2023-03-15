/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/logsumexp.cc
 * Max and sum of exponents of Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @date 2023-03-15
 * */

#include "nntile/tile/logsumexp.hh"
#include "nntile/starpu/logsumexp.hh"

namespace nntile
{
namespace tile
{

//! Tile-wise max and sum of exponents along single given axis
template<typename T>
void logsumexp_async(const Tile<T> &src, const Tile<T> &dst)
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
void logsumexp_async<fp64_t>(const Tile<fp64_t> &src, const Tile<fp64_t> &dst);

// Explicit instantiation
template
void logsumexp<fp32_t>(const Tile<fp32_t> &src, const Tile<fp32_t> &dst);

template
void logsumexp<fp64_t>(const Tile<fp64_t> &src, const Tile<fp64_t> &dst);

} // namespace tile
} // namespace nntile

