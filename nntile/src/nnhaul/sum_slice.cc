/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/sum_slice.cc
 * Sum over fibers into a slice of a StarPU buffer
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/sum_slice.hh"

// Standard libraries
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/sum_slice.hh"

namespace nntile::haul
{

//! Constructor
template<typename T>
SumSlice<std::tuple<T>>::SumSlice():
    codelet(
        "nntile_sum_slice",
        &SumSlice<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &SumSlice<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &SumSlice<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! StarPU wrapper for kernel::sum_slice::cpu<T>
template<typename T>
void SumSlice<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    const T *src = ::nntile::haul::buf_as<T>(buffers, 0);
    T *dst = ::nntile::haul::buf_as<T>(buffers, 1);
    // Launch kernel
    kernel::sum_slice::cpu<T>(args->m, args->n, args->k, args->alpha, src,
            args->beta, dst);
}

// Specializations of CPU wrapper for accelerated types
template<>
void SumSlice<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    SumSlice<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void SumSlice<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    SumSlice<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void SumSlice<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    SumSlice<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! StarPU wrapper for kernel::sum_slice::cuda<T>
template<typename T>
void SumSlice<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    const T *src = ::nntile::haul::buf_as<T>(buffers, 0);
    T *dst = ::nntile::haul::buf_as<T>(buffers, 1);
    // Get CUDA stream
    cudaStream_t stream = ::nnhaul::cuda_stream();
    // Launch kernel
    kernel::sum_slice::cuda<T>(stream, args->m, args->n, args->k, args->alpha,
            src, args->beta, dst);
}

// Specializations of CUDA wrapper for accelerated types
template<>
void SumSlice<std::tuple<fp32_fast_tf32_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    SumSlice<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void SumSlice<std::tuple<fp32_fast_fp16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    SumSlice<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void SumSlice<std::tuple<fp32_fast_bf16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    SumSlice<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for sum_slice tasks
template<typename T>
std::uint64_t SumSlice<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
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
void SumSlice<std::tuple<T>>::submit(int starpu_worker_hint, Index m, Index n, Index k, Scalar alpha, ::nnhaul::Handle & src, Scalar beta,
        ::nnhaul::Handle & dst, int redux)
//! Insert sum_slice task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * nntile_starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    // Access mode for the dst handle
    constexpr Scalar zero = 0, one = 1;
    enum starpu_data_access_mode dst_mode;
    if(beta == zero)
    {
        dst_mode = STARPU_W;
    }
    else if(beta == one)
    {
        if(redux != 0)
        {
            dst_mode = STARPU_REDUX;
        }
        else
        {
            dst_mode = static_cast<starpu_data_access_mode>(STARPU_RW | STARPU_COMMUTE);
        }
    }
    else
    {
        dst_mode = STARPU_RW;
    }
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
        { { STARPU_R, &src }, { dst_mode, &dst } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class SumSlice<std::tuple<nntile::fp64_t>>;
template class SumSlice<std::tuple<nntile::fp32_t>>;
template class SumSlice<std::tuple<nntile::fp32_fast_tf32_t>>;
template class SumSlice<std::tuple<nntile::fp32_fast_fp16_t>>;
template class SumSlice<std::tuple<nntile::fp32_fast_bf16_t>>;
template class SumSlice<std::tuple<nntile::bf16_t>>;
template class SumSlice<std::tuple<nntile::fp16_t>>;

//! Pack of sum_slice operations for different types
sum_slice_pack_t sum_slice;

} // namespace nntile::haul
