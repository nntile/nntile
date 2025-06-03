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

// Corresponding header
#include "nntile/starpu/embedding.hh"

// Standard libraries
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/embedding.hh"

namespace nntile::starpu
{

//! Constructor
template<typename T>
Embedding<std::tuple<T>>::Embedding():
    codelet("nntile_embedding", footprint, cpu_funcs, cuda_funcs)
{
    codelet.set_modes_fixed({STARPU_R, STARPU_R, STARPU_RW});
}

//! Apply embedding on StarPU buffer on CPU
template<typename T>
void Embedding<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    args_t *args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const int64_t *index = interfaces[0]->get_ptr<int64_t>();
    const T *vocab = interfaces[1]->get_ptr<T>();
    T *embed = interfaces[2]->get_ptr<T>();
    // Get embeddings
    kernel::embedding::cpu<T>(
        args->m,
        args->n,
        args->k,
        args->k_start,
        args->k_size,
        index,
        vocab,
        embed
    );
#endif // STARPU_SIMGRID
}

// Specializations of CPU wrapper for accelerated types
template<>
void Embedding<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Embedding<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Embedding<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Embedding<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Embedding<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Embedding<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! Apply embedding on StarPU buffer on CUDA
template<typename T>
void Embedding<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    args_t *args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const int64_t *index = interfaces[0]->get_ptr<int64_t>();
    const T *vocab = interfaces[1]->get_ptr<T>();
    T *embed = interfaces[2]->get_ptr<T>();
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Get embeddings
    kernel::embedding::cuda<T>(
        stream,
        args->m,
        args->n,
        args->k,
        args->k_start,
        args->k_size,
        index,
        vocab,
        embed
    );
#endif // STARPU_SIMGRID
};

// Specializations of CPU wrapper for accelerated types
template<>
void Embedding<std::tuple<fp32_fast_tf32_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Embedding<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void Embedding<std::tuple<fp32_fast_fp16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Embedding<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void Embedding<std::tuple<fp32_fast_bf16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Embedding<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for embedding tasks that depends only on cl_arg
template<typename T>
uint32_t Embedding<std::tuple<T>>::footprint(struct starpu_task *task)
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

template<typename T>
void Embedding<std::tuple<T>>::submit(Index m, Index n, Index k, Index k_start, Index k_size,
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
    int ret = starpu_task_insert(&codelet,
            STARPU_R, index.get(),
            STARPU_R, vocab.get(),
            STARPU_RW, embed.get(),
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
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class Embedding<std::tuple<nntile::fp64_t>>;
template class Embedding<std::tuple<nntile::fp32_t>>;
template class Embedding<std::tuple<nntile::fp32_fast_tf32_t>>;
template class Embedding<std::tuple<nntile::fp32_fast_fp16_t>>;
template class Embedding<std::tuple<nntile::fp32_fast_bf16_t>>;
template class Embedding<std::tuple<nntile::bf16_t>>;

//! Pack of embedding operations for different types
embedding_pack_t embedding;

} // namespace nntile::starpu
