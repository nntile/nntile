/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/cpu/scal.cc
 * Scaling operation for buffer
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-03
 * */

#include "nntile/kernel/cpu/scal.hh"
#include "nntile/kernel/args/scal.hh"
#include "nntile/starpu.hh"

namespace nntile
{

//! Scale buffer by a scalar
//
// X = alpha * X
//
// @param[in] nelems: Number of elements in a buffer
// @param[in] alpha: Scaling factor
// @param[inout] src: Buffer itself
template<typename T>
void scal_kernel_cpu(Index nelems, T alpha, T *src)
    noexcept
{
    for(Index i = 0; i < nelems; ++i)
    {
        src[i] *= alpha;
    }
}

//! Scale buffer by a scalar
//
// X = alpha * X
//
// @param[in] nelems: Number of elements in a buffer
// @param[in] alpha: Scaling factor
// @param[inout] src: Buffer itself
template<typename T>
void scal_starpu_cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<scal_starpu_args<T> *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<StarpuVariableInterface **>(buffers);
    // Launch kernel
    T *src = interfaces[0]->get_ptr<T>();
    scal_kernel_cpu<T>(args->nelems, args->alpha, src);
}

// Explicit instantiation
template
void scal_starpu_cpu<fp32_t>(void *buffers[], void *cl_args)
    noexcept;

template
void scal_starpu_cpu<fp64_t>(void *buffers[], void *cl_args)
    noexcept;

} // namespace nntile

