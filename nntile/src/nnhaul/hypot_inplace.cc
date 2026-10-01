/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/hypot_inplace.cc
 * hypot_inplace operation on a StarPU buffers
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/hypot_inplace.hh"

// Standard libraries
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/hypot_inplace.hh"
#include "nntile/nnhaul/ops/scale.hh"
#include "nntile/nnhaul/ops/clear.hh"
#include "nntile/nnhaul/ops/scale_inplace.hh"

namespace nntile::haul
{

//! Constructor
template<typename T>
HypotInplace<std::tuple<T>>::HypotInplace():
    codelet("nntile_hypot_inplace", &HypotInplace<std::tuple<T>>::cpu, nullptr, &HypotInplace<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! Apply hypot_inplace operation for StarPU buffers in CPU
template<typename T>
void HypotInplace<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    const T *src = ::nntile::haul::buf_as<T>(buffers, 0);
    T *dst = ::nntile::haul::buf_as<T>(buffers, 1);
    // Launch kernel
    kernel::hypot_inplace::cpu<T>(args->nelems, args->alpha, src, args->beta, dst);
}

// Specializations of CPU wrapper for accelerated types
template<>
void HypotInplace<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    HypotInplace<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void HypotInplace<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    HypotInplace<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void HypotInplace<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    HypotInplace<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}


//! Footprint for hypot_inplace tasks that depends only on cl_arg
template<typename T>
std::uint64_t HypotInplace<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(&args->nelems, sizeof(args->nelems), hash);
    return hash;
}

//! Submit hypot_inplace task
template<typename T>
void HypotInplace<std::tuple<T>>::submit(int starpu_worker_hint,
        Index nelems, Scalar alpha, ::nnhaul::Handle & src, Scalar beta, ::nnhaul::Handle & dst)
//! Insert hypot_inplace task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * nntile_starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    constexpr Scalar zero = 0, one = 1;
    // If beta is zero this function reduces to scale
    //if(beta == zero)
    //{
    //    throw std::runtime_error("modulus(apha*src) is not yet implemented");
    //    //scale::submit<T>(nelems, alpha, src, dst);
    //    return;
    //}
    // If beta is non-zero and alpha is zero then reduce to scale_inplace
    //if(alpha == zero)
    //{
    //    throw std::runtime_error("modulus_inplace(beta*dst) is not yet "
    //            "implemented");
    //    //scale_inplace::submit<T>(nelems, beta, dst);
    //    return;
    //}
    // Access mode for the dst handle
    enum starpu_data_access_mode dst_mode;
    if(beta == one)
    {
        dst_mode = static_cast<starpu_data_access_mode>(STARPU_RW | STARPU_COMMUTE);
    }
    else if(beta == zero)
    {
        dst_mode = STARPU_W;
    }
    else
    {
        dst_mode = STARPU_RW;
    }
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->nelems = nelems;
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
template class HypotInplace<std::tuple<nntile::fp64_t>>;
template class HypotInplace<std::tuple<nntile::fp32_t>>;
template class HypotInplace<std::tuple<nntile::fp32_fast_tf32_t>>;
template class HypotInplace<std::tuple<nntile::fp32_fast_fp16_t>>;
template class HypotInplace<std::tuple<nntile::fp32_fast_bf16_t>>;
template class HypotInplace<std::tuple<nntile::bf16_t>>;
template class HypotInplace<std::tuple<nntile::fp16_t>>;

//! Pack of hypot_inplace operations for different types
hypot_inplace_pack_t hypot_inplace;

} // namespace nntile::haul
