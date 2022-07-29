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
 * @date 2022-04-22
 * */

#include <nntile/kernel/cpu/scal.hh>
#include <starpu_data_interfaces.h>

namespace nntile
{

template<typename T>
void scal_kernel_cpu(Index nelems, T alpha, T *src)
    noexcept
{
    for(Index i = 0; i < nelems; ++i)
    {
        src[i] *= alpha;
    }
}

template<typename T>
void scal_starpu_cpu(void *buffers[], void *cl_args)
    noexcept
{
    Index nelems;
    T alpha;
    starpu_codelet_unpack_args(cl_args, &nelems, &alpha);
    T *src = reinterpret_cast<T *>(STARPU_NDIM_GET_PTR(buffers[0]));
    scal_kernel_cpu<T>(nelems, alpha, src);
}

// Explicit instantiation
template
void scal_starpu_cpu<fp32_t>(void *buffers[], void *cl_args)
    noexcept;

template
void scal_starpu_cpu<fp64_t>(void *buffers[], void *cl_args)
    noexcept;

} // namespace nntile

