/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/flash_maxsumexp.cc
 * Fused materialization and maxsumexp for StarPU buffer
 *
 * @version 1.1.0
 * */

#include "nntile/starpu/flash_maxsumexp.hh"
#ifndef STARPU_SIMGRID
#include "nntile/kernel/flash_maxsumexp.hh"
#endif // STARPU_SIMGRID
#include <cstdlib>
#include <cmath>
#include <limits>

namespace nntile::starpu::flash_maxsumexp
{

#ifdef NNTILE_USE_CBLAS
//! Rematerialize and compute maxsumexp along middle axis of StarPU buffer on CPU
template<typename T>
void cpu(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *K = interfaces[0]->get_ptr<T>();
    const T *Q = interfaces[1]->get_ptr<T>();
    const bool_t *mask = interfaces[2]->get_ptr<bool_t>();
    T *maxsumexp = interfaces[3]->get_ptr<T>();
    // Launch kernels (not yet implemented)
#endif // STARPU_SIMGRID
}
#endif // NNTILE_USE_CBLAS

#ifdef NNTILE_USE_CUDA
//! Max and sum of exponents along middle axis of StarPU buffer on CUDA
template<typename T>
void cuda(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *K = interfaces[0]->get_ptr<T>();
    const T *Q = interfaces[1]->get_ptr<T>();
    const bool_t *mask = interfaces[2]->get_ptr<bool_t>();
    T *maxsumexp = interfaces[3]->get_ptr<T>();
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Launch kernel (commented out code shows operations)
    // Index K_offset = args->head * args->seq;
    // Index Q_offset = K_offset;
    // Index tmp_offset = args->seq * args->seq;
    // cublas_batch(handle, CUBLAS_OP_T, CUBLAS_OP_N,
    //         args->seq, args->seq, args->head, 1.0/std::sqrt(args->head),
    //         K, args->head, K_offset, Q, args->head, Q_offset,
    //         0.0, tmp, args->seq, tmp_offset, args->batch);
    // using Y = typename T::repr_t;
    // kernel::mask_scalar::cuda<T>(stream, args->seq*args->seq, args->batch,
    //         mask, -std::numeric_limits<Y>::infinity(), tmp);
    // kernel::maxsumexp::cuda<T>(stream, 1, args->seq*args->batch, args->seq,
    //         tmp, maxsumexp);
    kernel::flash_maxsumexp::cuda<T>(stream, args->batch, args->seq, args->head,
            K, Q, mask, maxsumexp);
#endif // STARPU_SIMGRID
}
#endif // NNTILE_USE_CUDA

//! Footprint for maxsumexp tasks that depends only on m, n and k
static
uint32_t footprint(struct starpu_task *task)
{
    // Get arguments
    auto args = reinterpret_cast<args_t *>(task->cl_arg);
    // Apply hash over parameters m, n and k. This way if we swap values of m,
    // n and k, then the total size of buffers will remain the same, but the
    // footprint will be different
    uint32_t hash = 0;
    hash = starpu_hash_crc32c_be_n(&args->seq, sizeof(args->seq), hash);
    hash = starpu_hash_crc32c_be_n(&args->head, sizeof(args->head), hash);
    hash = starpu_hash_crc32c_be_n(&args->batch, sizeof(args->batch), hash);
    return hash;
}

Codelet codelet_fp32, codelet_fp64, codelet_fp32_fast_tf32, codelet_bf16,
        codelet_fp32_fast_fp16, codelet_fp32_fast_bf16;

void init()
{
    codelet_fp32.init("nntile_flash_maxsumexp_fp32",
            footprint,
#ifdef NNTILE_USE_CBLAS
            {}, // {cpu<fp32_t>},
#else // NNTILE_USE_CBLAS
            {},
#endif // NNTILE_USE_CBLAS
#ifdef NNTILE_USE_CUDA
            {cuda<fp32_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );

        codelet_bf16.init("nntile_flash_maxsumexp_bf16",
            footprint,
#ifdef NNTILE_USE_CBLAS
            {},
#else // NNTILE_USE_CBLAS
            {},
#endif // NNTILE_USE_CBLAS
#ifdef NNTILE_USE_CUDA
            {cuda<bf16_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );

    codelet_fp64.init("nntile_flash_maxsumexp_fp64",
            footprint,
#ifdef NNTILE_USE_CBLAS
            {}, // {cpu<fp64_t>},
#else // NNTILE_USE_CBLAS
            {},
#endif // NNTILE_USE_CBLAS
#ifdef NNTILE_USE_CUDA
            {cuda<fp64_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );

    codelet_fp32_fast_tf32.init("nntile_flash_maxsumexp_fp32_fast_tf32",
            footprint,
#ifdef NNTILE_USE_CBLAS
            {},
#else // NNTILE_USE_CBLAS
            {},
#endif // NNTILE_USE_CBLAS
#ifdef NNTILE_USE_CUDA
            {cuda<fp32_fast_tf32_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );

    codelet_fp32_fast_fp16.init("nntile_flash_maxsumexp_fp32_fast_fp16",
            footprint,
#ifdef NNTILE_USE_CBLAS
            {},
#else // NNTILE_USE_CBLAS
            {},
#endif // NNTILE_USE_CBLAS
#ifdef NNTILE_USE_CUDA
            {cuda<fp32_fast_fp16_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );

    codelet_fp32_fast_bf16.init("nntile_flash_maxsumexp_fp32_fast_bf16",
            footprint,
#ifdef NNTILE_USE_CBLAS
            {},
#else // NNTILE_USE_CBLAS
            {},
#endif // NNTILE_USE_CBLAS
#ifdef NNTILE_USE_CUDA
            {cuda<fp32_fast_bf16_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );
}

void restrict_where(uint32_t where)
{
    codelet_fp32.restrict_where(where);
    codelet_bf16.restrict_where(where);
    codelet_fp64.restrict_where(where);
    codelet_fp32_fast_tf32.restrict_where(where);
    codelet_fp32_fast_fp16.restrict_where(where);
    codelet_fp32_fast_bf16.restrict_where(where);
}

void restore_where()
{
    codelet_fp32.restore_where();
    codelet_bf16.restore_where();
    codelet_fp64.restore_where();
    codelet_fp32_fast_tf32.restore_where();
    codelet_fp32_fast_fp16.restore_where();
    codelet_fp32_fast_bf16.restore_where();
}

template<typename T>
void submit(Index seq, Index head, Index batch, Handle K, Handle Q,
        Handle mask, Handle maxsumexp, int redux)
//! Insert flash_maxsumexp task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->seq = seq;
    args->head = head;
    args->batch = batch;
    // Access mode for the maxsumexp handle
    enum starpu_data_access_mode maxsumexp_mode;
    if(redux != 0)
    {
        maxsumexp_mode = STARPU_REDUX;
        //maxsumexp_mode = Config::STARPU_RW_COMMUTE;
    }
    else
    {
        maxsumexp_mode = Config::STARPU_RW_COMMUTE;
    }
    // Submit task
    double nflops = 2 * seq * seq * head * batch;
    int ret = starpu_task_insert(codelet<T>(),
            STARPU_R, static_cast<starpu_data_handle_t>(K),
            STARPU_R, static_cast<starpu_data_handle_t>(Q),
            STARPU_R, static_cast<starpu_data_handle_t>(mask),
            maxsumexp_mode, static_cast<starpu_data_handle_t>(maxsumexp),
            STARPU_CL_ARGS, args, sizeof(*args),
            STARPU_FLOPS, nflops,
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in flash_maxsumexp task submission");
    }
}

// Explicit instantiation
template
void submit<fp32_t>(Index seq, Index head, Index batch, Handle K, Handle Q,
        Handle mask, Handle maxsumexp, int redux);

template
void submit<bf16_t>(Index seq, Index head, Index batch, Handle K, Handle Q,
        Handle mask, Handle maxsumexp, int redux);

template
void submit<fp32_fast_tf32_t>(Index seq, Index head, Index batch, Handle K, Handle Q,
        Handle mask, Handle maxsumexp, int redux);

template
void submit<fp32_fast_fp16_t>(Index seq, Index head, Index batch, Handle K, Handle Q,
        Handle mask, Handle maxsumexp, int redux);

template
void submit<fp32_fast_bf16_t>(Index seq, Index head, Index batch, Handle K, Handle Q,
        Handle mask, Handle maxsumexp, int redux);

template
void submit<fp64_t>(Index seq, Index head, Index batch, Handle K, Handle Q,
        Handle mask, Handle maxsumexp, int redux);

} // namespace nntile::starpu::flash_maxsumexp
