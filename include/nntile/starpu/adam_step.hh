/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu/adam_step.hh
 * Adam step with StarPU buffers
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @author Aleksandr Mikhalev
 * @date 2023-07-21
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/starpu/config.hh>

namespace nntile
{
namespace starpu
{
namespace adam_step
{

//! Structure for arguments
template<typename T>
struct args_t
{
    Index num_iter;
    Index num_elems;
    T beta_1;
    T beta_2;
    T eps;
    T lr;
};

// Apply Adam step to StarPU buffers on CPU
template<typename T>
void cpu(void *buffers[], void *cl_args)
    noexcept;

#ifdef NNTILE_USE_CUDA
// Apply Adam step of StarPU buffers on CUDA
template<typename T>
void cuda(void *buffers[], void *cl_args)
    noexcept;
#endif // NNTILE_USE_CUDA

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
void submit(Index num_iter, Index num_elems, T beta_1, T beta_2, T eps, T lr,
            Handle grad, Handle first_moment, Handle second_moment, Handle p);

} // namespace adam_step
} // namespace starpu
} // namespace nntile

