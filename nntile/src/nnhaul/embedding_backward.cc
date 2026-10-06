/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/embedding_backward.cc
 * Embeddings from vocabulary within StarPU buffers
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/embedding_backward.hh"

// Standard libraries
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/embedding_backward.hh"

namespace nntile::haul
{

//! Constructor
template<typename T>
EmbeddingBackward<std::tuple<T>>::EmbeddingBackward():
    codelet(
        "nntile_embedding_backward",
        &EmbeddingBackward<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &EmbeddingBackward<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &EmbeddingBackward<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! Apply embedding backward on StarPU buffer on CPU
template<typename T>
void EmbeddingBackward<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    args_t const *args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    const int64_t *index = ::nntile::haul::buf_as<int64_t>(buffers, 0);
    const T *embed = ::nntile::haul::buf_as<T>(buffers, 1);
    T *vocab = ::nntile::haul::buf_as<T>(buffers, 2);
    // Accumulate vocab gradients
    kernel::embedding_backward::cpu<T>(
        args->m,
        args->n,
        args->k,
        args->k_start,
        args->k_size,
        args->index_range,
        args->vocab_nelems,
        args->alpha,
        args->beta,
        index,
        embed,
        vocab
    );
}

// Specializations of CPU wrapper for accelerated types
template<>
void EmbeddingBackward<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    EmbeddingBackward<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void EmbeddingBackward<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    EmbeddingBackward<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void EmbeddingBackward<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    EmbeddingBackward<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! Apply embedding backward on StarPU buffer on CUDA
template<typename T>
void EmbeddingBackward<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    args_t const *args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    const int64_t *index = ::nntile::haul::buf_as<int64_t>(buffers, 0);
    const T *embed = ::nntile::haul::buf_as<T>(buffers, 1);
    T *vocab = ::nntile::haul::buf_as<T>(buffers, 2);
    // Get CUDA stream
    cudaStream_t stream = ::nnhaul::cuda_stream();
    // Accumulate vocab gradients
    kernel::embedding_backward::cuda<T>(
        stream,
        args->m,
        args->n,
        args->k,
        args->k_start,
        args->k_size,
        args->index_range,
        args->vocab_nelems,
        args->alpha,
        args->beta,
        index,
        embed,
        vocab
    );
}

// Specializations of CUDA wrapper for accelerated types
template<>
void EmbeddingBackward<std::tuple<fp32_fast_tf32_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    EmbeddingBackward<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void EmbeddingBackward<std::tuple<fp32_fast_fp16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    EmbeddingBackward<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void EmbeddingBackward<std::tuple<fp32_fast_bf16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    EmbeddingBackward<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for embedding tasks that depends only on cl_arg
template<typename T>
std::uint64_t EmbeddingBackward<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Apply hash over parameters m, n, k and k_size.
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(&args->m, sizeof(args->m), hash);
    hash = ::nntile::haul::fnv1a(&args->n, sizeof(args->n), hash);
    hash = ::nntile::haul::fnv1a(&args->k, sizeof(args->k), hash);
    hash = ::nntile::haul::fnv1a(&args->k_size, sizeof(args->k_size), hash);
    hash = ::nntile::haul::fnv1a(&args->index_range, sizeof(args->index_range), hash);
    hash = ::nntile::haul::fnv1a(&args->vocab_nelems, sizeof(args->vocab_nelems),
            hash);
    hash = ::nntile::haul::fnv1a(&args->alpha, sizeof(args->alpha), hash);
    hash = ::nntile::haul::fnv1a(&args->beta, sizeof(args->beta), hash);
    return hash;
}

template<typename T>
void EmbeddingBackward<std::tuple<T>>::submit(int starpu_worker_hint, Index m,
        Index n, Index k, Index k_start, Index k_size, Index index_range,
        Index vocab_nelems,
        Scalar alpha, Scalar beta, ::nnhaul::Handle & index, ::nnhaul::Handle & embed, ::nnhaul::Handle & vocab,
        int redux)
//! Insert embedding_backward task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * nntile_starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    constexpr Scalar zero = 0, one = 1;
    // Access mode for the output vocab handle
    enum starpu_data_access_mode vocab_mode;
    if(beta == zero)
    {
        vocab_mode = STARPU_W;
    }
    else if(beta == one)
    {
        if(redux != 0)
        {
            vocab_mode = STARPU_REDUX;
        }
        else
        {
            vocab_mode = static_cast<starpu_data_access_mode>(
                    STARPU_RW | STARPU_COMMUTE);
        }
    }
    else
    {
        throw std::runtime_error("embedding_backward: beta must be 0.0 or 1.0");
    }
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->m = m;
    args->n = n;
    args->k = k;
    args->k_start = k_start;
    args->k_size = k_size;
    args->index_range = index_range;
    args->vocab_nelems = vocab_nelems;
    args->alpha = alpha;
    args->beta = beta;
    double nflops = m * n * k_size;
    // Submit task
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &index }, { STARPU_R, &embed }, { vocab_mode, &vocab } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class EmbeddingBackward<std::tuple<nntile::fp64_t>>;
template class EmbeddingBackward<std::tuple<nntile::fp32_t>>;
template class EmbeddingBackward<std::tuple<nntile::fp32_fast_tf32_t>>;
template class EmbeddingBackward<std::tuple<nntile::fp32_fast_fp16_t>>;
template class EmbeddingBackward<std::tuple<nntile::fp32_fast_bf16_t>>;
template class EmbeddingBackward<std::tuple<nntile::bf16_t>>;
template class EmbeddingBackward<std::tuple<nntile::fp16_t>>;

//! Pack of embedding backward operations for different types
embedding_backward_pack_t embedding_backward;

} // namespace nntile::haul
