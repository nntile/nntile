/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/gelu/cpu.cc
 * GeLU operation on CPU
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/gelu/cpu.hh"
#include <cmath>
#include "nntile/kernel/cpu.hh"

namespace nntile::kernel::gelu
{

template<typename T>
void cpu(Index nelems, const T *src, T *dst)
    noexcept
//! GeLU operation performed on CPU
/*! Uses std::erfc() function, which implements the following:
 * GeLU(z) = 0.5 z erfc(-z/sqrt(2))
 *
 * @params[in] nelems: Number of elements in a buffer
 * @params[in] src: Input buffer to apply GeLU
 * @params[out] dst: Output buffer to apply GeLU
 * */
{
    using Y = typename T::repr_t;
    constexpr Y mone{-1.0}, pt5{0.5};
    const Y f1 = mone / std::sqrt(Y{2.0});
    for(Index i = 0; i < nelems; ++i)
    {
        Y z(src[i]);
        Y y = std::erfc(f1 * z);
        dst[i] = static_cast<T>((pt5 * z) * y);
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index nelems, const fp32_t *src, fp32_t *dst)
    noexcept;

template
void cpu<fp64_t>(Index nelems, const fp64_t *src, fp64_t *dst)
    noexcept;

template
void cpu<bf16_t>(Index nelems, const bf16_t *src, bf16_t *dst)
    noexcept;

template
void cpu<fp16_t>(Index nelems, const fp16_t *src, fp16_t *dst)
    noexcept;

} // namespace nntile::kernel::gelu
