/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/cpu/gelutanh.cc
 * Approximate GeLU operation on CPU based on tanh function
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-08
 * */

#include "nntile/kernel/cpu/gelutanh.hh"
#include "nntile/starpu.hh"
#include <cmath>

namespace nntile
{
namespace kernel
{
namespace cpu
{

//! Approximate GeLU operation inplace of a buffer on CPU
//
// GeLU(x) = 0.5x(1+tanh(sqrt(2/pi)(x+0.044715x^3)))
//
// @params[in] nelems: Number of elements in a buffer
// @params[inout] data: Buffer to apply GeLU
template<typename T>
void gelutanh(Index nelems, T *data)
    noexcept
{
    constexpr T pi = 3.141592653589793238462643383279502884L,
        pt5 = 0.5, f1 = T{0.044715};
    const T sqrt_pi = std::sqrt(pi), sqrt_2 = std::sqrt(T{2}),
        f2 = sqrt_2/sqrt_pi,
        f3 = -T{2}*f2, f4 = f3*f1,
        alpha = T{2}*f2, beta8 = T{8}*f1*f2;
    for(Index i = 0; i < nelems; ++i)
    {
        T x = data[i];
        //T y = x * pt5;
        //T tmp = alpha + beta8*y*y;
        //tmp = std::tanh(tmp*y);
        //data[i] = y*tmp + y;
        T z = x * (f3 + f4*x*x);
        data[i] = x / (T{1}+std::exp(z));
    }
}

// Explicit instantiation
template
void gelutanh<fp32_t>(Index nelems, fp32_t *data)
    noexcept;

template
void gelutanh<fp64_t>(Index nelems, fp64_t *data)
    noexcept;

} // namespace cpu
} // namespace kernel
} // namespace nntile

