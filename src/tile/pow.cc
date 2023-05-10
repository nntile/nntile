/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/pow.cc
 * Power operation for Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-18
 * */

#include "nntile/tile/pow.hh"
#include "nntile/starpu/pow.hh"

namespace nntile
{
namespace tile
{

//! Asynchronous tile-wise power operation
/*! @param[inout] A: Tile for the element-wise power operation
 * */
template<typename T>
void pow_async(T alpha, T exp, const Tile<T> &A)
{
    // Submit task without any arguments checked
    starpu::pow::submit<T>(A.nelems, alpha, exp, A);
}

//! Blocking version of tile-wise power operation
/*! @param[inout] A: Tile for the element-wise power operation
 * */
template<typename T>
void pow(T alpha, T exp, const Tile<T> &A)
{
    pow_async<T>(alpha, exp, A);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void pow_async<fp32_t>(fp32_t alpha, fp32_t exp, const Tile<fp32_t> &A);

template
void pow_async<fp64_t>(fp64_t alpha, fp64_t exp, const Tile<fp64_t> &A);

// Explicit instantiation
template
void pow<fp32_t>(fp32_t alpha, fp32_t exp, const Tile<fp32_t> &A);

template
void pow<fp64_t>(fp64_t alpha, fp64_t exp, const Tile<fp64_t> &A);

} // namespace tile
} // namespace nntile

