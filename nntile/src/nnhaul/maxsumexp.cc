/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/maxsumexp.cc
 * Max and sum of exponents for StarPU buffer
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/maxsumexp.hh"

// Standard libraries
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/maxsumexp.hh"

namespace nntile::haul
{

//! Constructor
template<typename T>
MaxSumExp<std::tuple<T>>::MaxSumExp():
    codelet("nntile_maxsumexp", &MaxSumExp<std::tuple<T>>::cpu, nullptr, &MaxSumExp<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! Max and sum of exponents along middle axis of StarPU buffer on CPU
template<typename T>
void MaxSumExp<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    const T *src = ::nntile::haul::buf_as<T>(buffers, 0);
    T *dst = ::nntile::haul::buf_as<T>(buffers, 1);
    // Launch kernel
    kernel::maxsumexp::cpu<T>(args->m, args->n, args->k, src, args->beta, dst);
}

// Specializations of CPU wrapper for accelerated types
template<>
void MaxSumExp<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    MaxSumExp<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void MaxSumExp<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    MaxSumExp<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void MaxSumExp<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    MaxSumExp<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}


//! Footprint for maxsumexp tasks that depends only on m, n and k
template<typename T>
std::uint64_t MaxSumExp<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(&args->m, sizeof(args->m), hash);
    hash = ::nntile::haul::fnv1a(&args->n, sizeof(args->n), hash);
    hash = ::nntile::haul::fnv1a(&args->k, sizeof(args->k), hash);
    hash = ::nntile::haul::fnv1a(&args->beta, sizeof(args->beta), hash);
    return hash;
}

template<typename T>
void MaxSumExp<std::tuple<T>>::submit(int starpu_worker_hint, Index m, Index n,
        Index k, ::nnhaul::Handle & src, ::nnhaul::Handle & dst, Scalar beta, int redux)
//! Insert maxsumexp task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * nntile_starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    constexpr Scalar zero = 0, one = 1;
    // Access mode for the dst handle
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
        throw std::runtime_error("maxsumexp: beta must be 0.0 or 1.0");
    }
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->m = m;
    args->n = n;
    args->k = k;
    args->beta = beta;
    // Put amount of bytes read and write inplace of gflops
    double nflops = sizeof(T) * m * (k+2) * n;
    // Submit task
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &src }, { dst_mode, &dst } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class MaxSumExp<std::tuple<nntile::fp64_t>>;
template class MaxSumExp<std::tuple<nntile::fp32_t>>;
template class MaxSumExp<std::tuple<nntile::fp32_fast_tf32_t>>;
template class MaxSumExp<std::tuple<nntile::fp32_fast_fp16_t>>;
template class MaxSumExp<std::tuple<nntile::fp32_fast_bf16_t>>;
template class MaxSumExp<std::tuple<nntile::bf16_t>>;
template class MaxSumExp<std::tuple<nntile::fp16_t>>;

//! Pack of maxsumexp operations for different types
maxsumexp_pack_t maxsumexp;

} // namespace nntile::haul
