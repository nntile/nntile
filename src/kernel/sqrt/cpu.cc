/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/sqrt/cpu.cc
 * Sqrt operation on CPU
 *
 * @version 1.0.0
 * */

#include "nntile/kernel/sqrt/cpu.hh"
#include <cmath>

namespace nntile::kernel::sqrt
{

template<typename T>
void cpu(Index nelems, const T *src, T *dst)
    noexcept
//! Sqrt operation on CPU
/*
 * @params[in] nelems: Number of elements in a buffer
 * @params[in] src: Input buffer to apply sqrt
 * @params[out] dst: Output buffer to apply sqrt
 * */
{
    for(Index i = 0; i < nelems; ++i)
    {
        dst[i] = std::sqrt(src[i]);
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index nelems, const fp32_t *src, fp32_t *dst)
    noexcept;

template
void cpu<fp64_t>(Index nelems, const fp64_t *src, fp64_t *dst)
    noexcept;

} // namespace nntile::kernel::sqrt

