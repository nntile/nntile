/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/addcdiv/cpu.cc
 * Per-element addcdiv operation for buffers on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @date 2023-02-14
 * */

#include "nntile/kernel/addcdiv/cpu.hh"

namespace nntile
{
namespace kernel
{
namespace addcdiv
{

template<typename T>
void cpu(T val, T eps, Index nelems, const T *nom, const T* denom, T *res)
    noexcept
//! Per-element addcdiv operation of buffers
/*! One of the buffers serves as output
 * 
 * @param[in] val: scalar that is multiplied on the division result
 * @param[in] eps: small scalar to avoid division by zero
 * @param[in] nelems: Number of elements in both buffers
 * @param[in] nom: Input buffer used as nominator
 * @param[in] denom: Input buffer used as denominator
 * @param[inout] res: Input buffers that contains output in the end
 * */
{
    // Cycle over buffers
    for(Index i = 0; i < nelems; ++i)
    {
        res[i] += val * nom[i] / (denom[i] + eps);
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(fp32_t val, fp32_t eps, Index nelems,
                 const fp32_t* nom, const fp32_t* denom, fp32_t* res)
    noexcept;

template
void cpu<fp64_t>(fp64_t val, fp64_t eps, Index nelems,
                 const fp64_t* nom, const fp64_t* denom, fp64_t* res)
    noexcept;

} // namespace addcdiv
} // namespace kernel
} // namespace nntile