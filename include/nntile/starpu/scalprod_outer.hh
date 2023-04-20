/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu/scalprod_outer.hh
 * Scalar products of two StarPU buffers along outer axes
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-20
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/starpu/config.hh>

namespace nntile
{
namespace starpu
{
namespace scalprod_outer
{

//! Structure for arguments
template<typename T>
struct args_t
{
    Index m;
    Index n;
    Index k;
    T alpha;
    T beta;
};

// Scalar products along outer axes of StarPU buffer on CPU
template<typename T>
void cpu(void *buffers[], void *cl_args)
    noexcept;

extern Codelet codelet_fp32, codelet_fp64;

template<typename T>
constexpr Codelet *codelet()
{
    throw std::runtime_error("Non-supported type");
    return nullptr;
}

template<>
constexpr Codelet *codelet<fp32_t>()
{
    return &codelet_fp32;
}

template<>
constexpr Codelet *codelet<fp64_t>()
{
    return &codelet_fp64;
}

void init();

void restrict_where(uint32_t where);

void restore_where();

template<typename T>
void submit(Index m, Index n, Index k, T alpha, Handle src1, Handle src2,
        T beta, Handle dst);

} // namespace scalprod_outer
} // namespace starpu
} // namespace nntile

