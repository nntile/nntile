/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/pow/cpu.cc
 * Power operation on CPU
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/pow/cpu.hh"
#include <cmath>
#include "nntile/kernel/cpu.hh"

namespace nntile::kernel::pow
{

template<typename T>
void cpu(Index nelems, Scalar alpha_, Scalar exp_, T *data_)
    noexcept
//! Inplace power operation on CPU
/*! Does the following per-element operation:
 * pow(z) = alpha * z^exp
 *
 * @params[in] nelems: Number of elements in a buffer
 * @params[in] alpha_: scalar multplier for output
 * @params[in] exp_: exponent parameter
 * @params[inout] data_: Buffer to apply power function
 * */
{
    using Y = typename CPUComputeType<T>::value;
    auto data = reinterpret_cast<Y *>(data_);
    const Y alpha{alpha_}, exp{exp_};
    for(Index i = 0; i < nelems; ++i)
    {
        Y z = data[i];
        if(exp == -1)
        {
            data[i] = alpha / z;
        }
        else
        {
            data[i] = alpha * std::pow(z, exp);
        }
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index nelems, Scalar alpha, Scalar exp, fp32_t *data)
    noexcept;

template
void cpu<fp64_t>(Index nelems, Scalar alpha, Scalar exp, fp64_t *data)
    noexcept;

} // namespace nntile::kernel::pow
