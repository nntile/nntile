/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
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
 * @date 2023-04-18
 * */

#include "nntile/tile/nrm2.hh"
#include "nntile/starpu/nrm2.hh"
#include "nntile/starpu/clear.hh"
#include "nntile/starpu/scal.hh"
#include "nntile/starpu/hypot.hh"

namespace nntile
{
namespace tile
{

//! Tile-wise Euclidian norm
template<typename T>
void nrm2_async(T alpha, const Tile<T> &src, T beta, const Tile<T> &dst,
        const Tile<T> &tmp)
{
    // Check dimensions
    if(dst.ndim != 0)
    {
        throw std::runtime_error("dst.ndim != 0");
    }
    if(tmp.ndim != 0)
    {
        throw std::runtime_error("tmp.ndim != 0");
    }
    // Insert task
    if(beta == 0.0)
    {
        if(alpha == 0.0)
        {
            starpu::clear::submit(dst);
        }
        else
        {
            starpu::nrm2::submit<T>(src.nelems, src, dst);
            if(alpha != 1.0)
            {
                starpu::scal::submit<T>(std::abs(alpha), dst.nelems, dst);
            }
        }
    }
    else
    {
        starpu::nrm2::submit<T>(src.nelems, src, tmp);
        starpu::hypot::submit<T>(alpha, tmp, beta, dst);
    }
}

//! Tile-wise Euclidian norm
template<typename T>
void nrm2(T alpha, const Tile<T> &src, T beta, const Tile<T> &dst,
        const Tile<T> &tmp)
{
    nrm2_async<T>(alpha, src, beta, dst, tmp);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void nrm2_async<fp32_t>(fp32_t alpha, const Tile<fp32_t> &src, fp32_t beta,
        const Tile<fp32_t> &dst, const Tile<fp32_t> &tmp);

template
void nrm2_async<fp64_t>(fp64_t alpha, const Tile<fp64_t> &src, fp64_t beta,
        const Tile<fp64_t> &dst, const Tile<fp64_t> &tmp);

// Explicit instantiation
template
void nrm2<fp32_t>(fp32_t alpha, const Tile<fp32_t> &src, fp32_t beta,
        const Tile<fp32_t> &dst, const Tile<fp32_t> &tmp);

template
void nrm2<fp64_t>(fp64_t alpha, const Tile<fp64_t> &src, fp64_t beta,
        const Tile<fp64_t> &dst, const Tile<fp64_t> &tmp);

} // namespace tile
} // namespace nntile

