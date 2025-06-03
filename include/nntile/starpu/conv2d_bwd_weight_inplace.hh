/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu/conv2d_bwd_weight_inplace.hh
 * Backward 2D-Convolution of two tensors in WHCN format to get grad of weight
 * Due to Fortran ordering, WHCN of NNTile is equal to NCHF format of PyTorch
 *
 * @version 1.1.0
 * */

#pragma once

// Compile-time definitions
#include <nntile/defs.h>

// Standard headers
#include <tuple>

// NNTile headers
#include <nntile/starpu/codelet.hh>
#include <nntile/starpu/handle.hh>

namespace nntile::starpu
{

//! Generic wrapper class for conv2d_bwd_weight_inplace operation is not defined
template<typename T>
class Conv2dBwdWeightInplace;

//! Specialization of wrapper class for conv2d_bwd_weight_inplace operation via std::tuple
template<typename T>
class Conv2dBwdWeightInplace<std::tuple<T>>
{
public:
    //! Codelet for the current operation
    CodeletTyped<T> codelet;

    //! Constructor
    Conv2dBwdWeightInplace();

    //! Structure for operation arguments
    struct args_t
    {
        Index src1_m;
        Index src1_n;
        Index src1_channels;
        Index batch;
        Index src2_m;
        Index src2_n;
        Index stride_m;
        Index stride_n;
        Index src2_channels;
        Index offset_m;
        Index offset_n;
        Scalar alpha;
        Index dst_m;
        Index dst_n;
        Index dilation_m;
        Index dilation_n;
        Scalar beta;
    };

    //! Footprint function for the current operation
    static uint32_t footprint(struct starpu_task *task);

    //! Wrapper for a generic CPU implementation
    static void cpu(void *buffers[], void *cl_args)
        noexcept;

    //! Array of all wrappers for CPU implementations
    static constexpr func_array cpu_funcs = {
        cpu
    };

#ifdef NNTILE_USE_CUDA
    //! Wrapper for a generic CUDA implementation
    static void cuda(void *buffers[], void *cl_args)
        noexcept;

    //! Array of all wrappers for CUDA implementations
    static constexpr func_array cuda_funcs = {
        cuda
    };
#else // NNTILE_USE_CUDA
    //! Array of all wrappers for CUDA implementations
    static constexpr func_array cuda_funcs = {};
#endif // NNTILE_USE_CUDA

    //! Submit add task
    void submit(
        Index src1_m,
        Index src1_n,
        Index src1_channels,
        Index batch,
        Index src2_m,
        Index src2_n,
        Index stride_m,
        Index stride_n,
        Index src2_channels,
        Index offset_m,
        Index offset_n,
        Scalar alpha,
        Handle src1,
        Handle src2,
        Index dst_m,
        Index dst_n,
        Index dilation_m,
        Index dilation_n,
        Scalar beta,
        Handle dst
    );
};

//! Pack of conv2d_bwd_weight_inplace operations for different types
using conv2d_bwd_weight_inplace_pack_t = OperationPack<
    Conv2dBwdWeightInplace,
    std::tuple<nntile::fp64_t>,
    std::tuple<nntile::fp32_t>,
    std::tuple<nntile::fp32_fast_tf32_t>,
    std::tuple<nntile::fp32_fast_fp16_t>,
    std::tuple<nntile::fp32_fast_bf16_t>,
    std::tuple<nntile::bf16_t>
>;

//! Pack of conv2d_bwd_weight_inplace operations for different types
extern conv2d_bwd_weight_inplace_pack_t conv2d_bwd_weight_inplace;

} // namespace nntile::starpu
