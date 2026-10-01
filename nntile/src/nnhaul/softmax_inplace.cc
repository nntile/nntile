/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/softmax_inplace.cc
 * Inplace softmax operation for StarPU buffer
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/softmax_inplace.hh"

// Standard libraries
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/softmax_inplace.hh"

namespace nntile::haul
{

//! Constructor
template<typename T>
SoftmaxInplace<std::tuple<T>>::SoftmaxInplace():
    codelet(
        "nntile_softmax_inplace",
        &SoftmaxInplace<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &SoftmaxInplace<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &SoftmaxInplace<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! StarPU wrapper for kernel::softmax_inplace::cpu<T>
template<typename T>
void SoftmaxInplace<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    const T *maxsumexp = ::nntile::haul::buf_as<T>(buffers, 0);
    T *dst = ::nntile::haul::buf_as<T>(buffers, 1);
    // Launch kernel
    kernel::softmax_inplace::cpu<T>(args->m, args->n, args->k, maxsumexp,
            args->alpha, dst);
}

// Specializations of CPU wrapper for accelerated types
template<>
void SoftmaxInplace<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    SoftmaxInplace<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void SoftmaxInplace<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    SoftmaxInplace<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void SoftmaxInplace<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    SoftmaxInplace<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! StarPU wrapper for kernel::softmax_inplace::cuda<T>
template<typename T>
void SoftmaxInplace<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    const T *maxsumexp = ::nntile::haul::buf_as<T>(buffers, 0);
    T *dst = ::nntile::haul::buf_as<T>(buffers, 1);
    // Get CUDA stream
    cudaStream_t stream = ::nnhaul::cuda_stream();
    // Launch kernel
    kernel::softmax_inplace::cuda<T>(stream, args->m, args->n, args->k,
            maxsumexp, args->alpha, dst);
}

// Specializations of CUDA wrapper for accelerated types
template<>
void SoftmaxInplace<std::tuple<fp32_fast_tf32_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    SoftmaxInplace<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void SoftmaxInplace<std::tuple<fp32_fast_fp16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    SoftmaxInplace<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void SoftmaxInplace<std::tuple<fp32_fast_bf16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    SoftmaxInplace<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for softmax_inplace tasks that depends only on m, n and k
template<typename T>
std::uint64_t SoftmaxInplace<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Apply hash over parameters m, n and k. This way if we swap values of m,
    // n and k, then the total size of buffers will remain the same, but the
    // footprint will be different
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(&args->m, sizeof(args->m), hash);
    hash = ::nntile::haul::fnv1a(&args->n, sizeof(args->n), hash);
    hash = ::nntile::haul::fnv1a(&args->k, sizeof(args->k), hash);
    return hash;
}

template<typename T>
void SoftmaxInplace<std::tuple<T>>::submit(int starpu_worker_hint, Index m, Index n, Index k, ::nnhaul::Handle & maxsumexp, Scalar alpha,
        ::nnhaul::Handle & dst)
//! Insert softmax_inplace task into StarPU pool of tasks
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
    args->alpha = alpha;
    // Put amount of bytes read and write inplace of gflops
    double nflops = sizeof(T) * m * (2*k+1) * n;
    // Submit task
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &maxsumexp }, { STARPU_RW, &dst } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class SoftmaxInplace<std::tuple<nntile::fp64_t>>;
template class SoftmaxInplace<std::tuple<nntile::fp32_t>>;
template class SoftmaxInplace<std::tuple<nntile::fp32_fast_tf32_t>>;
template class SoftmaxInplace<std::tuple<nntile::fp32_fast_fp16_t>>;
template class SoftmaxInplace<std::tuple<nntile::fp32_fast_bf16_t>>;
template class SoftmaxInplace<std::tuple<nntile::bf16_t>>;
template class SoftmaxInplace<std::tuple<nntile::fp16_t>>;

//! Pack of softmax_inplace operations for different types
softmax_inplace_pack_t softmax_inplace;

} // namespace nntile::haul
