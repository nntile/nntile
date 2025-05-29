/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu/clear.hh
 * Clear a StarPU buffer
 *
 * @version 1.1.0
 * */

#pragma once

// Compile-time definitions
#include <nntile/defs.h>

// NNTile headers
#include <nntile/starpu/codelet.hh>
#include <nntile/starpu/handle.hh>

namespace nntile::starpu::clear
{

//! Wrapper for all kernel functions
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
#else // NNTILE_USE_CUDA
    static constexpr func_array cuda_funcs = {};
#endif // NNTILE_USE_CUDA
};

// Declare codelet
extern Codelet codelet;

//! Insert task to clear buffer
void submit(
    Handle data
);

} // namespace nntile::starpu::clear
