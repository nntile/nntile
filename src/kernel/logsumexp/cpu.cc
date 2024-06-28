/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/logsumexp/cpu.cc
 * Logsumexp after computed maxsumexp result of a buffer on CPU
 *
 * @version 1.0.0
 * */

#include "nntile/kernel/logsumexp/cpu.hh"
#include <cmath>

namespace nntile::kernel::logsumexp
{

template<typename T>
void cpu(Index nelems, const T *maxsumexp, T *logsumexp)
    noexcept
{
    for(Index i = 0; i < nelems; ++i)
    {
        logsumexp[i] = maxsumexp[2*i] + std::log(maxsumexp[2*i+1]);
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index nelems, const fp32_t *maxsumexp, fp32_t *logsumexp)
    noexcept;

template
void cpu<fp64_t>(Index nelems, const fp64_t *maxsumexp, fp64_t *logsumexp)
    noexcept;

} // namespace nntile::kernel::logsumexp
