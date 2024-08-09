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
 * @version 1.1.0
 * */

#include "nntile/kernel/logsumexp/cpu.hh"
#include <cmath>
#include "nntile/kernel/cpu.hh"

namespace nntile::kernel::logsumexp
{

template<typename T>
void cpu(Index nelems, const T *maxsumexp_, T *logsumexp_)
    noexcept
{
    using Y = typename T::repr_t;
    Y maxsumexp_val_even{0.0};
    Y maxsumexp_val_odd{0.0};
    for(Index i = 0; i < nelems; ++i)
    {
        maxsumexp_val_even = static_cast<Y>(maxsumexp_[2*i]);
        maxsumexp_val_odd = static_cast<Y>(maxsumexp_[2*i+1]);
        logsumexp_[i] = static_cast<T>(maxsumexp_val_even + std::log(maxsumexp_val_odd));
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index nelems, const fp32_t *maxsumexp, fp32_t *logsumexp)
    noexcept;

template
void cpu<fp64_t>(Index nelems, const fp64_t *maxsumexp, fp64_t *logsumexp)
    noexcept;

template
void cpu<bf16_t>(Index nelems, const bf16_t *maxsumexp, bf16_t *logsumexp)
    noexcept;

} // namespace nntile::kernel::logsumexp
