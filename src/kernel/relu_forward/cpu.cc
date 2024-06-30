/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/relu_forward/cpu.cc
 * Forward ReLU operation on CPU
 *
 * @version 1.0.0
 * */

#include "nntile/kernel/relu_forward/cpu.hh"
#include <cmath>
#include "nntile/kernel/cpu.hh"

namespace nntile::kernel::relu_forward
{

template<typename T>
void cpu(Index nelems, const T *src_, T *dst_)
    noexcept
//! Forward ReLU operation on CPU
/*! Does the following per-element operation:
 * dst[i] = max(src[i], 0)
 *
 * @params[in] nelems: Number of elements in a buffer
 * @params[in] src_: Input array
 * @params[out] dst_: Output array
 * */
{
    using Y = typename CPUComputeType<T>::value;
    auto src = reinterpret_cast<const Y *>(src_);
    auto dst = reinterpret_cast<Y *>(dst_);
    constexpr Y zero{0.0};
    for(Index i = 0; i < nelems; ++i)
    {
        dst[i] = std::fmax(src[i], zero);
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index nelems, const fp32_t *src, fp32_t *dst)
    noexcept;

template
void cpu<fp64_t>(Index nelems, const fp64_t *src, fp64_t *dst)
    noexcept;

} // namespace nntile::kernel::relu_forward
