/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/norm_fiber_inplace.cc
 * Euclidean norms over slices into a fiber of a product of a StarPU buffer
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/norm_fiber_inplace.hh"

// Standard libraries
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/norm_fiber_inplace.hh"

namespace nntile::haul
{

//! Constructor
template<typename T>
NormFiberInplace<std::tuple<T>>::NormFiberInplace():
    codelet("nntile_norm_fiber_inplace", &NormFiberInplace<std::tuple<T>>::cpu, nullptr, &NormFiberInplace<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! StarPU wrapper for kernel::norm_fiber::cpu<T>
template<typename T>
void NormFiberInplace<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    const T *src = ::nntile::haul::buf_as<T>(buffers, 0);
    T *dst = ::nntile::haul::buf_as<T>(buffers, 1);
    // Launch kernel
    kernel::norm_fiber_inplace::cpu<T>(args->m, args->n, args->k, args->batch,
            args->alpha, src, args->beta, dst);
}

// Specializations of CPU wrapper for accelerated types
template<>
void NormFiberInplace<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    NormFiberInplace<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void NormFiberInplace<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    NormFiberInplace<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void NormFiberInplace<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    NormFiberInplace<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}


//! Footprint for norm_fiber tasks
template<typename T>
std::uint64_t NormFiberInplace<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Apply hash over parameters m, n and k
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(&args->m, sizeof(args->m), hash);
    hash = ::nntile::haul::fnv1a(&args->n, sizeof(args->n), hash);
    hash = ::nntile::haul::fnv1a(&args->k, sizeof(args->k), hash);
    hash = ::nntile::haul::fnv1a(&args->batch, sizeof(args->batch), hash);
    return hash;
}

template<typename T>
void NormFiberInplace<std::tuple<T>>::submit(int starpu_worker_hint, Index m, Index n, Index k, Index batch, Scalar alpha, ::nnhaul::Handle & src,
        Scalar beta, ::nnhaul::Handle & dst, int redux)
//! Insert norm_fiber_inplace task into StarPU pool of tasks
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
    args->batch = batch;
    args->alpha = alpha;
    args->beta = beta;
    // Submit task
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &src }, { dst_mode, &dst } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class NormFiberInplace<std::tuple<nntile::fp64_t>>;
template class NormFiberInplace<std::tuple<nntile::fp32_t>>;
template class NormFiberInplace<std::tuple<nntile::fp32_fast_tf32_t>>;
template class NormFiberInplace<std::tuple<nntile::fp32_fast_fp16_t>>;
template class NormFiberInplace<std::tuple<nntile::fp32_fast_bf16_t>>;
template class NormFiberInplace<std::tuple<nntile::bf16_t>>;

//! Pack of norm_fiber_inplace operations for different types
norm_fiber_inplace_pack_t norm_fiber_inplace;

} // namespace nntile::haul
