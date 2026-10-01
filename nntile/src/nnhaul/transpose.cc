/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/transpose.cc
 * Transpose operation for StarPU buffers
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/transpose.hh"

// Standard libraries
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/transpose.hh"

namespace nntile::haul
{

//! Constructor
template<typename T>
Transpose<std::tuple<T>>::Transpose():
    codelet(
        "nntile_transpose",
        &Transpose<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &Transpose<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &Transpose<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! StarPU wrapper for kernel::transpose::cpu<T>
template<typename T>
void Transpose<std::tuple<T>>::cpu(void *buffers[], void *cl_args) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    const T *src = ::nntile::haul::buf_as<T>(buffers, 0);
    T *dst = ::nntile::haul::buf_as<T>(buffers, 1);
    // Launch kernel
    kernel::transpose::cpu<T>(args->m, args->n, args->alpha, src, dst);
}

// Specializations of CPU wrapper for accelerated types
template<>
void Transpose<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Transpose<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Transpose<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Transpose<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Transpose<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Transpose<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! StarPU wrapper for kernel::transpose::cuda<T>
template<typename T>
void Transpose<std::tuple<T>>::cuda(void *buffers[], void *cl_args) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    const T *src = ::nntile::haul::buf_as<T>(buffers, 0);
    T *dst = ::nntile::haul::buf_as<T>(buffers, 1);
    // Get CUDA stream
    cudaStream_t stream = ::nnhaul::cuda_stream();
    // Launch kernel
    kernel::transpose::cuda<T>(
        stream,
        args->m,
        args->n,
        args->alpha,
        src,
        dst
    );
}

// Specializations of CUDA wrapper for accelerated types
template<>
void Transpose<std::tuple<fp32_fast_tf32_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    Transpose<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void Transpose<std::tuple<fp32_fast_fp16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    Transpose<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void Transpose<std::tuple<fp32_fast_bf16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    Transpose<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for transpose tasks
template<typename T>
std::uint64_t Transpose<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(&args->m, sizeof(args->m), hash);
    hash = ::nntile::haul::fnv1a(&args->n, sizeof(args->n), hash);
    return hash;
}

template<typename T>
void Transpose<std::tuple<T>>::submit(int starpu_worker_hint, Index m, Index n, Scalar alpha, ::nnhaul::Handle & src, ::nnhaul::Handle & dst)
//! Insert transpose task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * nntile_starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->m = m;
    args->n = n;
    args->alpha = alpha;
    // Put amount of read-write bytes into flop count
    double nflops = sizeof(T) * 2 * m * n;
    // Submit task
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &src }, { STARPU_W, &dst } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class Transpose<std::tuple<nntile::fp64_t>>;
template class Transpose<std::tuple<nntile::fp32_t>>;
template class Transpose<std::tuple<nntile::fp32_fast_tf32_t>>;
template class Transpose<std::tuple<nntile::fp32_fast_fp16_t>>;
template class Transpose<std::tuple<nntile::fp32_fast_bf16_t>>;
template class Transpose<std::tuple<nntile::bf16_t>>;
template class Transpose<std::tuple<nntile::fp16_t>>;

//! Pack of transpose operations for different types
transpose_pack_t transpose;

} // namespace nntile::haul
