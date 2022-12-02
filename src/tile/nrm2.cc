/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/nrm2.cc
 * Euclidian norm of Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-12-02
 * */

#include "nntile/tile/nrm2.hh"
#include "nntile/starpu/nrm2.hh"

namespace nntile
{
namespace tile
{

//! Tile-wise Euclidian norm
template<typename T>
void nrm2_async(const Tile<T> &src, const Tile<T> &dst)
{
    // Check dimensions
    if(dst.ndim != 0)
    {
        throw std::runtime_error("dst.ndim != 0");
    }
    // Insert task
    starpu::nrm2::submit<T>(src.nelems, src, dst);
}

//! Tile-wise Euclidian norm
template<typename T>
void nrm2(const Tile<T> &src, const Tile<T> &dst)
{
    nrm2_async<T>(src, dst);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void nrm2_async<fp32_t>(const Tile<fp32_t> &src, const Tile<fp32_t> &dst);

template
void nrm2_async<fp64_t>(const Tile<fp64_t> &src, const Tile<fp64_t> &dst);

// Explicit instantiation
template
void nrm2<fp32_t>(const Tile<fp32_t> &src, const Tile<fp32_t> &dst);

template
void nrm2<fp64_t>(const Tile<fp64_t> &src, const Tile<fp64_t> &dst);

} // namespace tile
} // namespace nntile

