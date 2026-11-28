/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on the NNHaul runtime system.
 *
 * @file src/nnhaul/pow.cc
 * Power operation on a NNHaul buffer
 *
 * @version 2.0.0
 * */
/*!
 * INTENTIONALLY DISABLED in the NNTile 2.0 transformer-only scope.
 * Kept in-tree for future re-enablement. No product path calls this code;
 * its tests are not registered while the family is in
 * NNTILE_DISABLED_KERNELS.
 */
// Corresponding header
#include "nntile/nnhaul/ops/pow.hh"

// Standard libraries
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/pow.hh"
#include <nntile/kernel_scope.hh>

namespace nntile::nnhaul
{

//! Constructor
template<typename T>
Pow<std::tuple<T>>::Pow():
    codelet(
        "nntile_pow",
        &Pow<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &Pow<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &Pow<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! NNHaul wrapper for kernel::pow::cpu<T>
template<typename T>
void Pow<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    args_t const *args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    auto data = ::nntile::nnhaul::buf_chunks<T>(buffers, 0);
    // Launch kernel per flat chunk piece (single piece if unchunked)
    ::nntile::nnhaul::for_each_flat_piece(
        args->nelems,
        [&](std::size_t n, T *p) {
            kernel::pow::cpu<T>(n, args->alpha, args->exp, p);
        },
        data);
}

// Specializations of CPU wrapper for accelerated types
template<>
void Pow<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Pow<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Pow<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Pow<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Pow<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Pow<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! NNHaul wrapper for kernel::pow::cuda<T>
template<typename T>
void Pow<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    args_t const *args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    auto data = ::nntile::nnhaul::buf_chunks<T>(buffers, 0);
    // Get CUDA stream
    cudaStream_t stream = ::nnhaul::cuda_stream();
    // Launch kernel per flat chunk piece (single piece if unchunked)
    ::nntile::nnhaul::for_each_flat_piece(
        args->nelems,
        [&](std::size_t n, T *p) {
            kernel::pow::cuda<T>(stream, n, args->alpha, args->exp, p);
        },
        data);
}

// Specializations of CUDA wrapper for accelerated types
template<>
void Pow<std::tuple<fp32_fast_tf32_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Pow<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void Pow<std::tuple<fp32_fast_fp16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Pow<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void Pow<std::tuple<fp32_fast_bf16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Pow<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for add tasks that depends only on cl_arg
template<typename T>
std::uint64_t Pow<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    uint32_t hash = 0;
    hash = ::nntile::nnhaul::fnv1a(&args->nelems, sizeof(args->nelems), hash);
    return hash;
}

template<typename T>
void Pow<std::tuple<T>>::submit(int worker_hint, Index nelems, Scalar alpha, Scalar exp, ::nnhaul::Handle & data)
//! Insert pow task into NNHaul task pool
/*! No argument checking is performed. All the inputs are packed and passed to
 * submit() call. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    ::nntile::require_kernel_family("pow");
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->nelems = nelems;
    args->alpha = alpha;
    args->exp = exp;
    // Submit task
    ::nntile::nnhaul::insert_task(codelet.raw, worker_hint,
        { { ACCESS_RW, &data } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class Pow<std::tuple<nntile::fp64_t>>;
template class Pow<std::tuple<nntile::fp32_t>>;
template class Pow<std::tuple<nntile::fp32_fast_tf32_t>>;
template class Pow<std::tuple<nntile::fp32_fast_fp16_t>>;
template class Pow<std::tuple<nntile::fp32_fast_bf16_t>>;
template class Pow<std::tuple<nntile::bf16_t>>;

//! Pack of pow operations for different types
pow_pack_t pow;

} // namespace nntile::nnhaul
