/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/norm_slice.cc
 * Euclidean norms of fibers into a slice of a StarPU buffer (out-of-place version)
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/norm_slice.hh"

// Standard libraries
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/norm_slice.hh"

namespace nntile::haul
{

//! Constructor
template<typename T>
NormSlice<std::tuple<T>>::NormSlice():
    codelet(
        "nntile_norm_slice",
        &NormSlice<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &NormSlice<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &NormSlice<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! StarPU wrapper for kernel::norm_slice::norm_slice<T>
template<typename T>
void NormSlice<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    const T *src1 = ::nntile::haul::buf_as<T>(buffers, 0);
    const T *src2 = ::nntile::haul::buf_as<T>(buffers, 1);
    T *dst = ::nntile::haul::buf_as<T>(buffers, 2);
    // Launch kernel
    kernel::norm_slice::cpu<T>(args->m, args->n, args->k, args->alpha, src1,
            args->beta, src2, dst);
}

// Specializations of CPU wrapper for accelerated types
template<>
void NormSlice<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    NormSlice<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void NormSlice<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    NormSlice<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void NormSlice<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    NormSlice<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! StarPU wrapper for kernel::norm_slice::norm_slice<T>
template<typename T>
void NormSlice<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    const T *src1 = ::nntile::haul::buf_as<T>(buffers, 0);
    const T *src2 = ::nntile::haul::buf_as<T>(buffers, 1);
    T *dst = ::nntile::haul::buf_as<T>(buffers, 2);
    // Get CUDA stream
    cudaStream_t stream = ::nnhaul::cuda_stream();
    // Launch kernel
    kernel::norm_slice::cuda<T>(stream, args->m, args->n, args->k,
            args->alpha, src1, args->beta, src2, dst);
}

// Specializations of CUDA wrapper for accelerated types
template<>
void NormSlice<std::tuple<fp32_fast_tf32_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    NormSlice<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void NormSlice<std::tuple<fp32_fast_fp16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    NormSlice<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void NormSlice<std::tuple<fp32_fast_bf16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    NormSlice<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for norm_slice tasks
template<typename T>
std::uint64_t NormSlice<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Apply hash over parameters m, n and k
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(&args->m, sizeof(args->m), hash);
    hash = ::nntile::haul::fnv1a(&args->n, sizeof(args->n), hash);
    hash = ::nntile::haul::fnv1a(&args->k, sizeof(args->k), hash);
    return hash;
}

template<typename T>
void NormSlice<std::tuple<T>>::submit(int starpu_worker_hint, Index m, Index n, Index k, Scalar alpha, ::nnhaul::Handle & src1, Scalar beta,
        ::nnhaul::Handle & src2, ::nnhaul::Handle & dst, int redux)
//! Insert norm_slice task into StarPU pool of tasks
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
    args->beta = beta;
    // Put amount of bytes read and write inplace of gflops
    size_t src_nbytes = sizeof(T) * m * k * n;
    size_t dst_nbytes = sizeof(T) * m * n;
    double nflops = beta == 0.0 ? src_nbytes + dst_nbytes :
        src_nbytes + 2*dst_nbytes;
    // Submit task
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &src1 }, { STARPU_R, &src2 }, { STARPU_W, &dst } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class NormSlice<std::tuple<nntile::fp64_t>>;
template class NormSlice<std::tuple<nntile::fp32_t>>;
template class NormSlice<std::tuple<nntile::fp32_fast_tf32_t>>;
template class NormSlice<std::tuple<nntile::fp32_fast_fp16_t>>;
template class NormSlice<std::tuple<nntile::fp32_fast_bf16_t>>;
template class NormSlice<std::tuple<nntile::bf16_t>>;
template class NormSlice<std::tuple<nntile::fp16_t>>;

//! Pack of norm_slice operations for different types
norm_slice_pack_t norm_slice;

} // namespace nntile::haul
