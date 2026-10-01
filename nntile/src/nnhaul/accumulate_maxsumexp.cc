/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/accumulate_maxsumexp.cc
 * Accumulate one StarPU maxsumexp buffer into another
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/accumulate_maxsumexp.hh"

// Standard libraries
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/accumulate_maxsumexp.hh"

namespace nntile::haul
{

//! Constructor
template<typename T>
AccumulateMaxSumExp<std::tuple<T>>::AccumulateMaxSumExp():
    codelet(
        "nntile_accumulate_maxsumexp",
        &AccumulateMaxSumExp<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &AccumulateMaxSumExp<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        nullptr)
{
    // Modes cannot be variable for accumulate_maxsumexp operation
    // Construct modes
    constexpr std::array<starpu_data_access_mode, 2> modes = {
        static_cast<starpu_data_access_mode>(STARPU_RW | STARPU_COMMUTE),
        STARPU_R
    };
    // Set modes
    codelet.set_modes_fixed(modes);
}

//! Apply accumulate_maxsumexp operation for StarPU buffers in CPU
template<typename T>
void AccumulateMaxSumExp<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get interfaces
    Index nelems = (*reinterpret_cast<std::size_t const *>(cl_args)) / sizeof(T) / 2;
    T *dst = ::nntile::haul::buf_as<T>(buffers, 0);
    const T *src = ::nntile::haul::buf_as<T>(buffers, 1);
    // Launch kernel
    kernel::accumulate_maxsumexp::cpu<T>(nelems, src, dst);
}

// Specializations of CPU wrapper for accelerated types
template<>
void AccumulateMaxSumExp<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AccumulateMaxSumExp<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void AccumulateMaxSumExp<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AccumulateMaxSumExp<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void AccumulateMaxSumExp<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AccumulateMaxSumExp<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}


#ifdef NNTILE_USE_CUDA
//! Apply accumulate_maxsumexp for StarPU buffers on CUDA
template<typename T>
void AccumulateMaxSumExp<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get interfaces
    Index nelems =
        (*reinterpret_cast<std::size_t const *>(cl_args)) / sizeof(T) / 2;
    T *dst = ::nntile::haul::buf_as<T>(buffers, 0);
    const T *src = ::nntile::haul::buf_as<T>(buffers, 1);
    // Get CUDA stream
    cudaStream_t stream = ::nnhaul::cuda_stream();
    // Launch kernel
    kernel::accumulate_maxsumexp::cuda<T>(stream, nelems, src, dst);
}

// Specializations of CUDA wrapper for accelerated types
template<>
void AccumulateMaxSumExp<std::tuple<fp32_fast_tf32_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    AccumulateMaxSumExp<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void AccumulateMaxSumExp<std::tuple<fp32_fast_fp16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    AccumulateMaxSumExp<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void AccumulateMaxSumExp<std::tuple<fp32_fast_bf16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    AccumulateMaxSumExp<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

template<typename T>
void AccumulateMaxSumExp<std::tuple<T>>::submit(int starpu_worker_hint, ::nnhaul::Handle & src, ::nnhaul::Handle & dst)
//! Insert accumulate_maxsumexp task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * nntile_starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    //double nflops;
    // Submit task
    std::size_t nnhaul_nbytes = dst.nbytes();
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_RW | STARPU_COMMUTE, &dst }, { STARPU_R, &src } }, &nnhaul_nbytes, sizeof(nnhaul_nbytes));

}


// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class AccumulateMaxSumExp<std::tuple<nntile::fp64_t>>;
template class AccumulateMaxSumExp<std::tuple<nntile::fp32_t>>;
template class AccumulateMaxSumExp<std::tuple<nntile::fp32_fast_tf32_t>>;
template class AccumulateMaxSumExp<std::tuple<nntile::fp32_fast_fp16_t>>;
template class AccumulateMaxSumExp<std::tuple<nntile::fp32_fast_bf16_t>>;
template class AccumulateMaxSumExp<std::tuple<nntile::bf16_t>>;

//! Pack of accumulate_maxsumexp operations for different types
accumulate_maxsumexp_pack_t accumulate_maxsumexp;

} // namespace nntile::haul
