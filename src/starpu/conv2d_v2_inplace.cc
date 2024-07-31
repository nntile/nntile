/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/conv2d_v2_inplace.cc
 * StarPU wrappers for 2D-Convolution between 2 matrices
 *
 * @version 1.0.0
 * */

#include "nntile/starpu/conv2d_v2_inplace.hh"
#ifndef STARPU_SIMGRID
#   include "nntile/kernel/conv2d_v2_inplace.hh"
#endif // STARPU_SIMGRID
#include <cstdlib>

//! StarPU wrappers for conv2d_v2_inplace operation
namespace nntile::starpu::conv2d_v2_inplace
{

//! StarPU wrapper for kernel::conv2d_v2_inplace::cpu<T>
template <typename T> void cpu(void *buffers[], void *cl_args) noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *src = interfaces[0]->get_ptr<T>();
    const T *kernel = interfaces[1]->get_ptr<T>();
    T *dst = interfaces[2]->get_ptr<T>();
    // Launch kernel
    kernel::conv2d_v2_inplace::cpu<T>(args->src_m, args->src_n,
            args->in_channels, args->batch, args->offset_m, args->offset_n,
            args->alpha, src, args->kernel_m, args->kernel_n,
            args->out_channels, args->kernel_transpose, kernel, args->dst_m,
            args->dst_n, args->beta, dst);
#endif // STARPU_SIMGRID
}

//! Footprint for conv2d_v2_inplace tasks
static uint32_t footprint(struct starpu_task *task)
{
    // Get arguments
    auto args = reinterpret_cast<args_t *>(task->cl_arg);
    // Apply hash over parameters m, n and k
    uint32_t hash = 0;
    hash = starpu_hash_crc32c_be_n(args, sizeof(*args), hash);
    return hash;
}

Codelet codelet_bf16, codelet_fp32, codelet_fp32_fast_tf32, codelet_fp64;

void init()
{
    codelet_bf16.init("nntile_conv2d_v2_inplace_bf16",
            footprint,
            {cpu<bf16_t>},
            {}
            );
    codelet_fp32.init("nntile_conv2d_v2_inplace_fp32",
            footprint,
            {cpu<fp32_t>},
            {}
            );
    codelet_fp32_fast_tf32.init("nntile_conv2d_v2_inplace_fp32_fast_tf32",
            footprint,
            {cpu<fp32_fast_tf32_t>},
            {}
            );
    codelet_fp64.init("nntile_conv2d_v2_inplace_fp64",
            footprint,
            {cpu<fp64_t>},
            {}
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
void submit(Index src_m, Index src_n, Index in_channels, Index batch,
        Index offset_m, Index offset_n, Scalar alpha, Handle src,
        Index kernel_m, Index kernel_n, Index out_channels,
        bool kernel_transpose, Handle kernel, Index dst_m, Index dst_n,
        Scalar beta, Handle dst)
//! Insert conv2d_v2 task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->src_m = src_m;
    args->src_n = src_n;
    args->in_channels = in_channels;
    args->batch = batch;
    args->offset_m = offset_m;
    args->offset_n = offset_n;
    args->alpha = alpha;
    args->kernel_m = kernel_m;
    args->kernel_n = kernel_n;
    args->out_channels = out_channels;
    args->kernel_transpose = kernel_transpose;
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
            STARPU_R, static_cast<starpu_data_handle_t>(src),
            STARPU_R, static_cast<starpu_data_handle_t>(kernel),
            STARPU_CL_ARGS, args, sizeof(*args),
            dst_mode, static_cast<starpu_data_handle_t>(dst),
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in conv2d_v2_inplace task submission");
    }
}

// Explicit instantiation
template
void submit<bf16_t>(Index src_m, Index src_n, Index in_channels, Index batch,
        Index offset_m, Index offset_n, Scalar alpha, Handle src,
        Index kernel_m, Index kernel_n, Index out_channels,
        bool kernel_transpose, Handle kernel, Index dst_m, Index dst_n,
        Scalar beta, Handle dst);

template
void submit<fp32_t>(Index src_m, Index src_n, Index in_channels, Index batch,
        Index offset_m, Index offset_n, Scalar alpha, Handle src,
        Index kernel_m, Index kernel_n, Index out_channels,
        bool kernel_transpose, Handle kernel, Index dst_m, Index dst_n,
        Scalar beta, Handle dst);

template
void submit<fp32_fast_tf32_t>(Index src_m, Index src_n, Index in_channels,
        Index batch, Index offset_m, Index offset_n, Scalar alpha, Handle src,
        Index kernel_m, Index kernel_n, Index out_channels,
        bool kernel_transpose, Handle kernel, Index dst_m, Index dst_n,
        Scalar beta, Handle dst);

template
void submit<fp64_t>(Index src_m, Index src_n, Index in_channels, Index batch,
        Index offset_m, Index offset_n, Scalar alpha, Handle src,
        Index kernel_m, Index kernel_n, Index out_channels,
        bool kernel_transpose, Handle kernel, Index dst_m, Index dst_n,
        Scalar beta, Handle dst);

} // namespace nntile::starpu::conv2d_v2_inplace
