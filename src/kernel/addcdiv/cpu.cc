/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/addcdiv/cpu.cc
 * Per-element addcdiv operation for buffers on CPU
 *
 * @version 1.0.0
 * */

#include "nntile/kernel/addcdiv/cpu.hh"
#include "nntile/kernel/cpu.hh"

namespace nntile::kernel::addcdiv
{

template<typename T>
void cpu(T val_, T eps_, Index nelems, const T *nom_, const T* denom_,
        T *res_)
    noexcept
//! Per-element addcdiv operation of buffers
/*! One of the buffers serves as output
 * 
 * @param[in] val_: scalar that is multiplied on the division result
 * @param[in] eps_: small scalar to avoid division by zero
 * @param[in] nelems: Number of elements in both buffers
 * @param[in] nom_: Input buffer used as nominator
 * @param[in] denom_: Input buffer used as denominator
 * @param[inout] res_: Input buffers that contains output in the end
 * */
{
    using Y = typename CPUComputeType<T>::value;
    auto nom = reinterpret_cast<const Y *>(nom_);
    auto denom = reinterpret_cast<const Y *>(denom_);
    auto res = reinterpret_cast<Y *>(res_);
    const Y val{val_}, eps{eps_};
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

} // namespace nntile::kernel::addcdiv
