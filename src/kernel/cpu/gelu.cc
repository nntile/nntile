/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/cpu/gelu.cc
 * GeLU operation on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-03
 * */

#include "nntile/kernel/cpu/gelu.hh"
#include "nntile/starpu.hh"
#include <cmath>

namespace nntile
{

//! GeLU operation inplace of a buffer on CPU
//
// @params[in] nelems: Number of elements in a buffer
// @params[inout] data: Buffer to apply GeLU
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

//! GeLU operation inplace of a StarPU buffer on CPU
template<typename T>
void gelu_starpu_cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    Index nelems = reinterpret_cast<Index *>(cl_args)[0];
    // Get interfaces
    auto interfaces = reinterpret_cast<StarpuVariableInterface **>(buffers);
    // Launch kernel
    T *data = interfaces[0]->get_ptr<T>();
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

