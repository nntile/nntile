/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/nrm2.cc
 * Euclidean norm of Tile<T>
 *
 * @version 1.1.0
 * */

#include <cmath>

#include "nntile/tile/nrm2.hh"
#include "nntile/starpu/nrm2.hh"
#include "nntile/starpu/clear.hh"
#include "nntile/starpu/scal_inplace.hh"
#include "nntile/starpu/hypot.hh"
#include <cmath>

namespace nntile::tile
{

//! Tile-wise Euclidean norm
template<typename T>
void nrm2_async(Scalar alpha, const Tile<T> &src, Scalar beta, const Tile<T> &dst,
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
                starpu::scal_inplace::submit<T>(dst.nelems, std::fabs(alpha),
                        dst);
            }
        }
    }
    else
    {
        starpu::nrm2::submit<T>(src.nelems, src, tmp);
        starpu::hypot::submit<T>(1, alpha, tmp, beta, dst);
    }
}

//! Tile-wise Euclidean norm
template<typename T>
void nrm2(Scalar alpha, const Tile<T> &src, Scalar beta, const Tile<T> &dst,
        const Tile<T> &tmp)
{
    nrm2_async<T>(alpha, src, beta, dst, tmp);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void nrm2_async<fp32_t>(Scalar alpha, const Tile<fp32_t> &src, Scalar beta,
        const Tile<fp32_t> &dst, const Tile<fp32_t> &tmp);

template
void nrm2_async<fp64_t>(Scalar alpha, const Tile<fp64_t> &src, Scalar beta,
        const Tile<fp64_t> &dst, const Tile<fp64_t> &tmp);

// Explicit instantiation
template
void nrm2<fp32_t>(Scalar alpha, const Tile<fp32_t> &src, Scalar beta,
        const Tile<fp32_t> &dst, const Tile<fp32_t> &tmp);

template
void nrm2<fp64_t>(Scalar alpha, const Tile<fp64_t> &src, Scalar beta,
        const Tile<fp64_t> &dst, const Tile<fp64_t> &tmp);

} // namespace nntile::tile
