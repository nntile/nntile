/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/logsumexp/cpu.cc
 * Logsumexp after computed maxsumexp result of a buffer on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @date 2023-03-06
 * */

#include "nntile/kernel/logsumexp/cpu.hh"
#include <cmath>

namespace nntile
{
namespace kernel
{
namespace logsumexp
{

template<typename T>
void cpu(Index m, const T *maxsumexp, T *logsumexp)
    noexcept
{
    for (Index i = 0; i < m; ++i) 
    {
        logsumexp[i] = maxsumexp[2*i] + std::log(maxsumexp[2*i+1]);
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index m, const fp32_t *src, fp32_t *maxsumexp)
    noexcept;

template
void cpu<fp64_t>(Index m, const fp64_t *src, fp64_t *maxsumexp)
    noexcept;

} // namespace logsumexp
} // namespace kernel
} // namespace nntile