/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file src/kernel/sqrt/cpu.cc
 * Sqrt operation on CPU
 *
 * @version 2.0.0
 * */
/*!
 * INTENTIONALLY DISABLED in the NNTile 2.0 transformer-only scope.
 * Kept in-tree for future re-enablement. No product path calls this code;
 * its tests are not registered while the family is in
 * NNTILE_DISABLED_KERNELS.
 */
#include "nntile/kernel/sqrt/cpu.hh"
#include <cmath>
#include "nntile/kernel/cpu.hh"

namespace nntile::kernel::sqrt
{

template<typename T>
void cpu(Index nelems, const T *src, T *dst)
    noexcept
//! Sqrt operation on CPU
/*
 * @params[in] nelems: Number of elements in a buffer
 * @params[in] src_: Input buffer to apply sqrt
 * @params[out] dst_: Output buffer to apply sqrt
 * */
{
    using Y = typename T::repr_t;
    for(Index i = 0; i < nelems; ++i)
    {
        dst[i] = T{std::sqrt(Y{src[i]})};
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

} // namespace nntile::kernel::sqrt
