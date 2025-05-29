/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu/accumulate_maxsumexp.hh
 * Accumulate one StarPU maxsumexp buffer into another
 *
 * @version 1.1.0
 * */

#pragma once

// Compile-time definitions
#include <nntile/defs.h>

// NNTile headers
#include <nntile/starpu/codelet.hh>
#include <nntile/starpu/handle.hh>

namespace nntile::starpu::accumulate_maxsumexp
{

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
#else // NNTILE_USE_CUDA
    static constexpr func_array cuda_funcs = {};
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

//! Submit accumulate_maxsumexp task
template<typename T>
void submit(
    Handle src,
    Handle dst
);

} // namespace nntile::starpu::accumulate_maxsumexp
