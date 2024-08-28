/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/embedding.cc
 * Embeddings from vocabulary within StarPU buffers
 *
 * @version 1.1.0
 * */

#ifndef STARPU_SIMGRID
#include "nntile/kernel/embedding.hh"
#endif // STARPU_SIMGRID
#include "nntile/starpu/embedding.hh"

namespace nntile::starpu::embedding
{

//! Copy embedding from vocabulary within StarPU buffers on CPU
template<typename T>
void cpu(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const int64_t *index = interfaces[0]->get_ptr<int64_t>();
    const T *vocab = interfaces[1]->get_ptr<T>();
    T *embed = interfaces[2]->get_ptr<T>();
    // Get embeddings
    kernel::embedding::cpu<T>(args->m, args->n, args->k, args->k_start,
            args->k_size, index, vocab, embed);
#endif // STARPU_SIMGRID
}

#ifdef NNTILE_USE_CUDA
//! Copy embedding from vocabulary within StarPU buffers on CUDA
template<typename T>
void cuda(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const int64_t *index = interfaces[0]->get_ptr<int64_t>();
    const T *vocab = interfaces[1]->get_ptr<T>();
    T *embed = interfaces[2]->get_ptr<T>();
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Get embeddings
    kernel::embedding::cuda<T>(stream, args->m, args->n, args->k,
            args->k_start, args->k_size, index, vocab, embed);
#endif // STARPU_SIMGRID
}
#endif // NNTILE_USE_CUDA

//! Footprint for embedding tasks that depends only on cl_arg
static
uint32_t footprint(struct starpu_task *task)
{
    // Get arguments
    auto args = reinterpret_cast<args_t *>(task->cl_arg);
    // Apply hash over parameters m, n, k and k_size.
    uint32_t hash = 0;
    hash = starpu_hash_crc32c_be_n(&args->m, sizeof(args->m), hash);
    hash = starpu_hash_crc32c_be_n(&args->n, sizeof(args->n), hash);
    hash = starpu_hash_crc32c_be_n(&args->k, sizeof(args->k), hash);
    hash = starpu_hash_crc32c_be_n(&args->k_size, sizeof(args->k_size), hash);
    return hash;
}

Codelet codelet_fp32, codelet_fp64, codelet_fp32_fast_tf32, codelet_bf16;

void init()
{
    codelet_fp32.init("nntile_embedding_fp32",
            footprint,
            {cpu<fp32_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<fp32_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );

    codelet_bf16.init("nntile_embedding_bf16",
            footprint,
            {cpu<bf16_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<bf16_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );

    codelet_fp32_fast_tf32.init("nntile_embedding_fp32_fast_tf32",
            footprint,
            {cpu<fp32_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<fp32_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );

    codelet_fp64.init("nntile_embedding_fp64",
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
    codelet_fp64.restrict_where(where);
}

void restore_where()
{
    codelet_fp32.restore_where();
    codelet_bf16.restore_where();
    codelet_fp32_fast_tf32.restore_where();
    codelet_fp64.restore_where();
}

template<typename T>
void submit(Index m, Index n, Index k, Index k_start, Index k_size,
        Handle index, Handle vocab, Handle embed)
//! Insert embedding task into StarPU pool of tasks
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
    args->k_start = k_start;
    args->k_size = k_size;
    double nflops = m * n * k_size;
    // Submit task
    int ret = starpu_task_insert(codelet<T>(),
            STARPU_R, static_cast<starpu_data_handle_t>(index),
            STARPU_R, static_cast<starpu_data_handle_t>(vocab),
            STARPU_RW, static_cast<starpu_data_handle_t>(embed),
            //Config::STARPU_RW_COMMUTE, static_cast<starpu_data_handle_t>(embed),
            STARPU_CL_ARGS, args, sizeof(*args),
            STARPU_FLOPS, nflops,
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in embedding task submission");
    }
}

// Explicit instantiation
template
void submit<fp32_t>(Index m, Index n, Index k, Index k_start, Index k_size,
        Handle index, Handle vocab, Handle embed);

template
void submit<bf16_t>(Index m, Index n, Index k, Index k_start, Index k_size,
        Handle index, Handle vocab, Handle embed);

template
void submit<fp32_fast_tf32_t>(Index m, Index n, Index k, Index k_start, Index k_size,
        Handle index, Handle vocab, Handle embed);

template
void submit<fp64_t>(Index m, Index n, Index k, Index k_start, Index k_size,
        Handle index, Handle vocab, Handle embed);

} // namespace nntile::starpu::embedding
