/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/fill/cpu.cc
 * Fill operation on CPU
 *
 * @version 1.0.0
 * */

#include "nntile/kernel/fill/cpu.hh"
#include "nntile/kernel/cpu.hh"

namespace nntile::kernel::fill
{

template<typename T>
void cpu(Index nelems, T val_, T *data_)
    noexcept
//! Fill operation on CPU
/*! Sets all elements to the provided value
 * @params[in] nelems: Number of elements in a buffer
 * @param[in] val_: Input value
 * @params[out] data_: Output buffer
 * */
{
    using Y = typename CPUComputeType<T>::value;
    auto *data = reinterpret_cast<Y *>(data_);
    const Y val{val_};
    for(Index i = 0; i < nelems; ++i)
    {
        data[i] = val;
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index nelems, fp32_t val, fp32_t *data)
    noexcept;

template
void cpu<fp64_t>(Index nelems, fp64_t val, fp64_t *data)
    noexcept;

} // namespace nntile::kernel::fill
