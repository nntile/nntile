/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu/add_scalar.hh
 * Add_scalar operation on StarPU buffers
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/starpu/config.hh>
#include <nntile/defs.h>

namespace nntile::starpu::add_scalar
{

//! Structure for arguments
struct args_t
{
    args_t(Index num_elements_, Scalar alpha_, Scalar beta_) :
        num_elements(num_elements_),
        alpha(alpha_),
        beta(beta_)
        {
        }
    Index num_elements;
    Scalar alpha;
    Scalar beta;
};

// Apply add_scalar for StarPU buffer on CPU
template<typename T>
void cpu(void *buffers[], void *cl_args)
    noexcept;

#ifdef NNTILE_USE_CUDA
// Apply add_scalar of StarPU buffer on CUDA
// template<typename T>
// void cuda(void *buffers[], void *cl_args)
//     noexcept;
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
void submit(Index num_elements, Scalar alpha, Scalar beta, Handle dst);

} // namespace nntile::starpu::add_scalar
