/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/scalprod_outer/cpu.hh
 * Scalar product of buffers on CPU along outer axes
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-20
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile
{
namespace kernel
{
namespace scalprod_outer
{

template<typename T>
void cpu(Index m, Index n, Index k, T alpha, const T *src1, const T *src2,
        T beta, T *sumnorm)
    noexcept;

} // namespace scalprod_outer
} // namespace kernel
} // namespace nntile

