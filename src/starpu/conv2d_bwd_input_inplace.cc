/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/conv2d_bwd_input_inplace.cc
 * Backward 2D-Convolution of two tensors in WHCN format to get grad of input
 * Due to Fortran ordering, WHCN of NNTile is equal to NCHF format of PyTorch
 *
 * @version 1.1.0
 * */

#include "nntile/starpu/conv2d_bwd_input_inplace.hh"
#ifndef STARPU_SIMGRID
#   include "nntile/kernel/conv2d_bwd_input_inplace.hh"
#endif // STARPU_SIMGRID
#include <cstdlib>

//! StarPU wrappers for conv2d_bwd_input_inplace operation
namespace nntile::starpu::conv2d_bwd_input_inplace
{

//! StarPU wrapper for kernel::conv2d_bwd_input_inplace::cpu<T>
template <typename T>
void cpu(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *src1 = interfaces[0]->get_ptr<T>();
    const T *src2 = interfaces[1]->get_ptr<T>();
    T *dst = interfaces[2]->get_ptr<T>();
    // Launch kernel
    kernel::conv2d_bwd_input_inplace::cpu<T>(args->src1_m, args->src1_n,
            args->stride_m, args->stride_n, args->src1_channels, args->batch,
            args->src2_m, args->src2_n, args->dilation_m, args->dilation_n,
            args->dst_channels, args->offset_m, args->offset_n, args->alpha,
            src1, src2, args->dst_m, args->dst_n, args->beta, dst);
#endif // STARPU_SIMGRID
}

#ifdef NNTILE_USE_CUDA
//! StarPU wrapper for kernel::conv2d_bwd_input_inplace::cuda<T>
template <typename T>
void cuda(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *src1 = interfaces[0]->get_ptr<T>();
    const T *src2 = interfaces[1]->get_ptr<T>();
    T *dst = interfaces[2]->get_ptr<T>();
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Launch kernel
    kernel::conv2d_bwd_input_inplace::cuda<T>(stream, args->src1_m,
            args->src1_n, args->stride_m, args->stride_n, args->src1_channels,
            args->batch, args->src2_m, args->src2_n, args->dilation_m,
            args->dilation_n, args->dst_channels, args->offset_m,
            args->offset_n, args->alpha, src1, src2, args->dst_m, args->dst_n,
            args->beta, dst);
#endif // STARPU_SIMGRID
}
#endif // NNTILE_USE_CUDA

//! Footprint for conv2d_bwd_input_inplace tasks
static uint32_t footprint(struct starpu_task *task)
{
    // Get arguments
    auto args = reinterpret_cast<args_t *>(task->cl_arg);
    // Apply hash over entire args
    uint32_t hash = 0;
    hash = starpu_hash_crc32c_be_n(args, sizeof(*args), hash);
    return hash;
}

Codelet codelet_bf16, codelet_fp32, codelet_fp32_fast_tf32, codelet_fp64;

void init()
{
    codelet_bf16.init("nntile_conv2d_bwd_input_inplace_bf16",
            footprint,
            {cpu<bf16_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<bf16_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );
    codelet_fp32.init("nntile_conv2d_bwd_input_inplace_fp32",
            footprint,
            {cpu<fp32_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<fp32_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );
    codelet_fp32_fast_tf32.init(
            "nntile_conv2d_bwd_input_inplace_fp32_fast_tf32",
            footprint,
            {cpu<fp32_fast_tf32_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<fp32_fast_tf32_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );
    codelet_fp64.init("nntile_conv2d_bwd_input_inplace_fp64",
            footprint,
            {cpu<fp64_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<fp64_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );
}

void restrict_where(uint32_t where)
{
    codelet_bf16.restrict_where(where);
    codelet_fp32.restrict_where(where);
    codelet_fp32_fast_tf32.restrict_where(where);
    codelet_fp64.restrict_where(where);
}

void restore_where()
{
    codelet_bf16.restore_where();
    codelet_fp32.restore_where();
    codelet_fp32_fast_tf32.restore_where();
    codelet_fp64.restore_where();
}

template <typename T>
void submit(Index src1_m, Index src1_n, Index stride_m, Index stride_n,
        Index src1_channels, Index batch, Index src2_m, Index src2_n,
        Index dilation_m, Index dilation_n, Index dst_channels, Index offset_m,
        Index offset_n, Scalar alpha, Handle src1, Handle src2, Index dst_m,
        Index dst_n, Scalar beta, Handle dst)
//! Insert conv2d_bwd_input_inplace task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->src1_m = src1_m;
    args->src1_n = src1_n;
    args->stride_m = stride_m;
    args->stride_n = stride_n;
    args->src1_channels = src1_channels;
    args->batch = batch;
    args->src2_m = src2_m;
    args->src2_n = src2_n;
    args->dilation_m = dilation_m;
    args->dilation_n = dilation_n;
    args->dst_channels = dst_channels;
    args->offset_m = offset_m;
    args->offset_n = offset_n;
    args->alpha = alpha;
    args->dst_m = dst_m;
    args->dst_n = dst_n;
    args->beta = beta;
    enum starpu_data_access_mode dst_mode = STARPU_RW;
    if(beta == 0.0)
    {
        dst_mode = STARPU_W;
    }
    // Submit task
    int ret = starpu_task_insert(codelet<T>(),
            STARPU_R, static_cast<starpu_data_handle_t>(src1),
            STARPU_R, static_cast<starpu_data_handle_t>(src2),
            STARPU_CL_ARGS, args, sizeof(*args),
            dst_mode, static_cast<starpu_data_handle_t>(dst),
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error(
                "Error in conv2d_bwd_input_inplace task submission");
    }
}

// Explicit instantiation
template
void submit<bf16_t>(Index src1_m, Index src1_n, Index stride_m, Index stride_n,
        Index src1_channels, Index batch, Index src2_m, Index src2_n,
        Index dilation_m, Index dilation_n, Index dst_channels, Index offset_m,
        Index offset_n, Scalar alpha, Handle src1, Handle src2, Index dst_m,
        Index dst_n, Scalar beta, Handle dst);

template
void submit<fp32_t>(Index src1_m, Index src1_n, Index stride_m, Index stride_n,
        Index src1_channels, Index batch, Index src2_m, Index src2_n,
        Index dilation_m, Index dilation_n, Index dst_channels, Index offset_m,
        Index offset_n, Scalar alpha, Handle src1, Handle src2, Index dst_m,
        Index dst_n, Scalar beta, Handle dst);

template
void submit<fp32_fast_tf32_t>(Index src1_m, Index src1_n, Index stride_m,
        Index stride_n, Index src1_channels, Index batch, Index src2_m,
        Index src2_n, Index dilation_m, Index dilation_n, Index dst_channels,
        Index offset_m, Index offset_n, Scalar alpha, Handle src1, Handle src2,
        Index dst_m, Index dst_n, Scalar beta, Handle dst);

template
void submit<fp64_t>(Index src1_m, Index src1_n, Index stride_m, Index stride_n,
        Index src1_channels, Index batch, Index src2_m, Index src2_n,
        Index dilation_m, Index dilation_n, Index dst_channels, Index offset_m,
        Index offset_n, Scalar alpha, Handle src1, Handle src2, Index dst_m,
        Index dst_n, Scalar beta, Handle dst);

} // namespace nntile::starpu::conv2d_bwd_input_inplace
