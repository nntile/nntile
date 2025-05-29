/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu/adam_step.hh
 * Adam step with StarPU buffers
 *
 * @version 1.1.0
 * */

#pragma once

// Compile-time definitions
#include <nntile/defs.h>

// NNTile headers
#include <nntile/starpu/codelet.hh>
#include <nntile/starpu/handle.hh>

namespace nntile::starpu::adam_step
{

//! Structure for arguments
struct args_t
{
    Index num_iter;
    Index num_elems;
    Scalar beta_1;
    Scalar beta_2;
    Scalar eps;
    Scalar lr;
    Scalar weight_decay;
};

//! Wrapper for all kernel functions
template<typename T>
struct KernelWrapper
{
    static void cpu(void *buffers[], void *cl_args)
        noexcept;

    static constexpr func_array cpu_funcs = {
        cpu
    };

#ifdef NNTILE_USE_CUDA
    static void cuda(void *buffers[], void *cl_args)
        noexcept;

    static constexpr func_array cuda_funcs = {
        cuda
    };
#endif // NNTILE_USE_CUDA
};

//! Codelet pack type for the current operation
using codelet_pack_t = CodeletPack<
    KernelWrapper,
    nntile::fp64_t,
    nntile::fp32_t,
    nntile::fp32_fast_tf32_t,
    nntile::fp32_fast_fp16_t,
    nntile::fp32_fast_bf16_t,
    nntile::bf16_t
>;

// Declare codelet pack
extern codelet_pack_t codelet_pack;

//! Submit Adam step task
template<typename T>
void submit(
    Index num_iter,
    Index num_elems,
    Scalar beta_1,
    Scalar beta_2,
    Scalar eps,
    Scalar lr,
    Scalar weight_decay,
    Handle grad,
    Handle first_moment,
    Handle second_moment,
    Handle param
);

} // namespace nntile::starpu::adam_step
