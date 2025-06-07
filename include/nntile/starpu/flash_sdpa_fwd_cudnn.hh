/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu/flash_sdpa_fwd_cudnn.hh
 * Cudnn forward pass for SDPA
 *
 * TODO: This file is yet under development.
 * @version 1.1.0
 * */

#pragma once

// Compile-time definitions
#include <nntile/defs.h>

// NNTile headers
#include <nntile/starpu/codelet.hh>
#include <nntile/starpu/handle.hh>

namespace nntile::starpu::flash_sdpa_fwd_cudnn
{

//! Structure for arguments
struct args_t
{
    Index seq;
    Index head;
    Index batch;
};

//! Wrapper for all kernel functions
template<typename T>
struct KernelWrapper
{
    static constexpr func_array cpu_funcs = {};

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
/*! No FP64, FP32 or accelerated FP32 types are supported due to cuDNN
 * limitations. */
using codelet_pack_t = CodeletPack<
    KernelWrapper,
    nntile::bf16_t
>;

// Declare codelet pack
extern codelet_pack_t codelet_pack;

//! Submit flash_sdpa_fwd_cudnn task
template<typename T>
void submit(
    Index seq,
    Index head,
    Index batch,
    Handle K,
    Handle Q,
    Handle mask,
    Handle logsumexp,
    Handle V,
    Handle A,
    int redux=0
);

} // namespace nntile::starpu::flash_sdpa_fwd_cudnn
