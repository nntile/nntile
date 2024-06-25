/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/rope3/cpu.hh
 * ROtary Positional Embedding
 *
 * @version 1.0.0
 * @author Gleb Karpov
 * @date 2024-06-15
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile
{
namespace kernel
{
namespace rope3
{

// Fill embedding from vocabulary
template<typename T>
void cpu(Index m, Index n, Index b, Index s, const T *sin, const T *cos,
        const T *src, T *dst)
    noexcept;

} // namespace rope3
} // namespace kernel
} // namespace nntile