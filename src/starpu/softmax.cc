/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/softmax.cc
 * Softmax operation for StarPU buffer
 *
 * @version 1.1.0
 * */

#ifndef STARPU_SIMGRID
#include "nntile/kernel/softmax.hh"
#endif // STARPU_SIMGRID
#include "nntile/starpu/softmax.hh"
#include <cstdlib>

namespace nntile::starpu::softmax
{

//! Softmax buffer along middle axis of StarPU buffer
template<typename T>
void cpu(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *maxsumexp = interfaces[0]->get_ptr<T>();
    const T *src = interfaces[1]->get_ptr<T>();
    T *dst = interfaces[2]->get_ptr<T>();
    // Launch kernel
    kernel::softmax::cpu<T>(args->m, args->n, args->k, maxsumexp, src,
            args->alpha, dst);
#endif // STARPU_SIMGRID
}

#ifdef NNTILE_USE_CUDA
//! Softmax buffer along middle axis of StarPU buffer
template<typename T>
void cuda(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *maxsumexp = interfaces[0]->get_ptr<T>();
    const T *src = interfaces[1]->get_ptr<T>();
    T *dst = interfaces[2]->get_ptr<T>();
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Launch kernel
    kernel::softmax::cuda<T>(stream, args->m, args->n, args->k, maxsumexp,
            src, args->alpha, dst);
#endif // STARPU_SIMGRID
}
#endif // NNTILE_USE_CUDA

//! Footprint for softmax tasks that depends only on m, n and k
static
uint32_t footprint(struct starpu_task *task)
{
    // Get arguments
    auto args = reinterpret_cast<args_t *>(task->cl_arg);
    // Apply hash over parameters m, n and k. This way if we swap values of m,
    // n and k, then the total size of buffers will remain the same, but the
    // footprint will be different
    uint32_t hash = 0;
    hash = starpu_hash_crc32c_be_n(&args->m, sizeof(args->m), hash);
    hash = starpu_hash_crc32c_be_n(&args->n, sizeof(args->n), hash);
    hash = starpu_hash_crc32c_be_n(&args->k, sizeof(args->k), hash);
    return hash;
}

Codelet codelet_fp32, codelet_fp64, codelet_fp32_fast_tf32,
        codelet_bf16, codelet_fp32_fast_fp16, codelet_fp32_fast_bf16;

void init()
{
    codelet_fp32.init("nntile_softmax_fp32",
            footprint,
            {cpu<fp32_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<fp32_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );

    codelet_bf16.init("nntile_softmax_bf16",
            footprint,
            {cpu<bf16_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<bf16_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );

    codelet_fp32_fast_tf32.init("nntile_softmax_fp32_fast_tf32",
            footprint,
            {cpu<fp32_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<fp32_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );

    codelet_fp32_fast_fp16.init("nntile_softmax_fp32_fast_fp16",
            footprint,
            {cpu<fp32_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<fp32_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );

    codelet_fp32_fast_bf16.init("nntile_softmax_fp32_fast_bf16",
            footprint,
            {cpu<fp32_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<fp32_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );

    codelet_fp64.init("nntile_softmax_fp64",
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
    codelet_fp32.restrict_where(where);
    codelet_bf16.restrict_where(where);
    codelet_fp32_fast_tf32.restrict_where(where);
    codelet_fp32_fast_fp16.restrict_where(where);
    codelet_fp32_fast_bf16.restrict_where(where);
    codelet_fp64.restrict_where(where);
}

void restore_where()
{
    codelet_fp32.restore_where();
    codelet_bf16.restore_where();
    codelet_fp32_fast_tf32.restore_where();
    codelet_fp32_fast_fp16.restore_where();
    codelet_fp32_fast_bf16.restore_where();
    codelet_fp64.restore_where();
}

template<typename T>
void submit(Index m, Index n, Index k, Handle maxsumexp, Handle src, Scalar alpha,
        Handle dst)
//! Insert softmax task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->m = m;
    args->n = n;
    args->k = k;
    args->alpha = alpha;
    // Put amount of bytes read and write inplace of gflops
    double nflops = sizeof(T) * m * (2*k+1) * n;
    // Submit task
    int ret = starpu_task_insert(codelet<T>(),
            STARPU_R, static_cast<starpu_data_handle_t>(maxsumexp),
            STARPU_R, static_cast<starpu_data_handle_t>(src),
            STARPU_W, static_cast<starpu_data_handle_t>(dst),
            STARPU_CL_ARGS, args, sizeof(*args),
            STARPU_FLOPS, nflops,
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in softmax task submission");
    }
}

// Explicit instantiation
template
void submit<fp32_t>(Index m, Index n, Index k, Handle maxsumexp, Handle src,
        Scalar alpha, Handle dst);

template
void submit<fp32_fast_tf32_t>(Index m, Index n, Index k, Handle maxsumexp, Handle src,
        Scalar alpha, Handle dst);

template
void submit<fp32_fast_fp16_t>(Index m, Index n, Index k, Handle maxsumexp, Handle src,
        Scalar alpha, Handle dst);

template
void submit<fp32_fast_bf16_t>(Index m, Index n, Index k, Handle maxsumexp, Handle src,
        Scalar alpha, Handle dst);

template
void submit<fp64_t>(Index m, Index n, Index k, Handle maxsumexp, Handle src,
        Scalar alpha, Handle dst);

template
void submit<bf16_t>(Index m, Index n, Index k, Handle maxsumexp, Handle src,
        Scalar alpha, Handle dst);

} // namespace nntile::starpu::softmax
