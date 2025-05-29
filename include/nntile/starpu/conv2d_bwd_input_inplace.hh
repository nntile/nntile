/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu/conv2d_bwd_input_inplace.hh
 * Backward 2D-Convolution of two tensors in WHCN format to get grad of input
 * Due to Fortran ordering, WHCN of NNTile is equal to NCHF format of PyTorch
 *
 * @version 1.1.0
 * */

#pragma once

// Compile-time definitions
#include <nntile/defs.h>

// NNTile headers
#include <nntile/starpu/codelet.hh>
#include <nntile/starpu/handle.hh>

namespace nntile::starpu::conv2d_bwd_input_inplace
{

//! Structure for arguments
struct args_t
{
    Index src1_m;
    Index src1_n;
    Index stride_m;
    Index stride_n;
    Index src1_channels;
    Index batch;
    Index src2_m;
    Index src2_n;
    Index dilation_m;
    Index dilation_n;
    Index dst_channels;
    Index offset_m;
    Index offset_n;
    Scalar alpha;
    Index dst_m;
    Index dst_n;
    Scalar beta;
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

//! Submit conv2d_bwd_input_inplace task
template <typename T>
void submit(
    Index src1_m,
    Index src1_n,
    Index stride_m,
    Index stride_n,
    Index src1_channels,
    Index batch,
    Index src2_m,
    Index src2_n,
    Index dilation_m,
    Index dilation_n,
    Index dst_channels,
    Index offset_m,
    Index offset_n,
    Scalar alpha,
    Handle src1,
    Handle src2,
    Index dst_m,
    Index dst_n,
    Scalar beta,
    Handle dst
);

} // namespace nntile::starpu::conv2d_bwd_input_inplace
