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
#include "nntile/nnhaul/ops/embedding.hh"

// Standard libraries
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/embedding.hh"

namespace nntile::haul
{

//! Constructor
template<typename T>
Embedding<std::tuple<T>>::Embedding():
    codelet("nntile_embedding", &Embedding<std::tuple<T>>::cpu, nullptr, &Embedding<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! Apply embedding on StarPU buffer on CPU
template<typename T>
void Embedding<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    args_t const *args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    const int64_t *index = ::nntile::haul::buf_as<int64_t>(buffers, 0);
    const T *vocab = ::nntile::haul::buf_as<T>(buffers, 1);
    T *embed = ::nntile::haul::buf_as<T>(buffers, 2);
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


//! Footprint for embedding tasks that depends only on cl_arg
template<typename T>
std::uint64_t Embedding<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Apply hash over parameters m, n, k and k_size.
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(&args->m, sizeof(args->m), hash);
    hash = ::nntile::haul::fnv1a(&args->n, sizeof(args->n), hash);
    hash = ::nntile::haul::fnv1a(&args->k, sizeof(args->k), hash);
    hash = ::nntile::haul::fnv1a(&args->k_size, sizeof(args->k_size), hash);
    return hash;
}

template<typename T>
void Embedding<std::tuple<T>>::submit(int starpu_worker_hint, Index m, Index n, Index k, Index k_start, Index k_size,
        ::nnhaul::Handle & index, ::nnhaul::Handle & vocab, ::nnhaul::Handle & embed)
//! Insert embedding task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * nntile_starpu_task_insert() function. If task submission fails, this routines
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
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &index }, { STARPU_R, &vocab }, { STARPU_RW, &embed } }, args, sizeof(*args));
    std::free(args);

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
template class Embedding<std::tuple<nntile::fp16_t>>;

//! Pack of embedding operations for different types
embedding_pack_t embedding;

} // namespace nntile::haul
