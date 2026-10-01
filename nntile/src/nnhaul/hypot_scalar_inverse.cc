/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/hypot_scalar_inverse.cc
 * Inverse of a hypot operation of a buffer and a scalar
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/hypot_scalar_inverse.hh"

// Standard libraries
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/hypot_scalar_inverse.hh"

namespace nntile::haul
{

//! Constructor
template<typename T>
HypotScalarInverse<std::tuple<T>>::HypotScalarInverse():
    codelet("nntile_hypot_scalar_inverse", &HypotScalarInverse<std::tuple<T>>::cpu, nullptr, &HypotScalarInverse<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! Apply hypot_scalar_inverse operation for StarPU buffers in CPU
template<typename T>
void HypotScalarInverse<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    T *dst = ::nntile::haul::buf_as<T>(buffers, 0);
    // Launch kernel
    kernel::hypot_scalar_inverse::cpu<T>(args->nelems, args->eps, args->alpha,
            dst);
}

// Specializations of CPU wrapper for accelerated types
template<>
void HypotScalarInverse<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    HypotScalarInverse<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void HypotScalarInverse<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    HypotScalarInverse<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void HypotScalarInverse<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    HypotScalarInverse<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}


//! Footprint for add tasks that depends only on cl_arg
template<typename T>
std::uint64_t HypotScalarInverse<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(&args->nelems, sizeof(args->nelems), hash);
    return hash;
}

//! Submit hypot_scalar_inverse task
template<typename T>
void HypotScalarInverse<std::tuple<T>>::submit(int starpu_worker_hint,
        Index nelems, Scalar eps, Scalar alpha, ::nnhaul::Handle & dst)
//! Insert hypot_scalar_inverse task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * nntile_starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->nelems = nelems;
    args->eps = eps;
    args->alpha = alpha;
    // Submit task
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_RW, &dst } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class HypotScalarInverse<std::tuple<nntile::fp64_t>>;
template class HypotScalarInverse<std::tuple<nntile::fp32_t>>;
template class HypotScalarInverse<std::tuple<nntile::fp32_fast_tf32_t>>;
template class HypotScalarInverse<std::tuple<nntile::fp32_fast_fp16_t>>;
template class HypotScalarInverse<std::tuple<nntile::fp32_fast_bf16_t>>;
template class HypotScalarInverse<std::tuple<nntile::bf16_t>>;
template class HypotScalarInverse<std::tuple<nntile::fp16_t>>;

//! Pack of hypot_scalar_inverse operations for different types
hypot_scalar_inverse_pack_t hypot_scalar_inverse;

} // namespace nntile::haul
