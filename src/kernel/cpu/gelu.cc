/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/cpu/gelu.cc
 * GeLU operation
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#include "nntile/kernel/cpu/gelu.hh"
#include <cmath>
#include <starpu_data_interfaces.h>

namespace nntile
{

// GeLU operation on a buffer
template<typename T>
void gelu_kernel_cpu(Index nelems, T *data)
    noexcept
{
    constexpr T one = 1, pt5 = 0.5;
    const T sqrt2 = std::sqrt(T{2.0});
    for(Index i = 0; i < nelems; ++i)
    {
        T tmp = pt5*(std::erf(data[i]/sqrt2)) + pt5;
        data[i] *= tmp;
    }
}

// GeLU operation on a StarPU buffer
template<typename T>
void gelu_starpu_cpu(void *buffers[], void *cl_args)
    noexcept
{
    Index nelems;
    starpu_codelet_unpack_args(cl_args, &nelems);
    T *data = reinterpret_cast<T *>(STARPU_NDIM_GET_PTR(buffers[0]));
    gelu_kernel_cpu<T>(nelems, data);
}

// Explicit instantiation
template
void gelu_starpu_cpu<fp32_t>(void *buffers[], void *cl_args)
    noexcept;

template
void gelu_starpu_cpu<fp64_t>(void *buffers[], void *cl_args)
    noexcept;

} // namespace nntile

