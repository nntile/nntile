/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/pow.cc
 * Power operation on a StarPU buffer
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/pow.hh"

// Standard libraries
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/pow.hh"

namespace nntile::haul
{

//! Constructor
template<typename T>
Pow<std::tuple<T>>::Pow():
    codelet("nntile_pow", &Pow<std::tuple<T>>::cpu, nullptr, &Pow<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! StarPU wrapper for kernel::pow::cpu<T>
template<typename T>
void Pow<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    args_t const *args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    T *data = ::nntile::haul::buf_as<T>(buffers, 0);
    // Launch kernel
    kernel::pow::cpu<T>(args->nelems, args->alpha, args->exp, data);
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


//! Footprint for add tasks that depends only on cl_arg
template<typename T>
std::uint64_t Pow<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(&args->nelems, sizeof(args->nelems), hash);
    return hash;
}

template<typename T>
void Pow<std::tuple<T>>::submit(int starpu_worker_hint, Index nelems, Scalar alpha, Scalar exp, ::nnhaul::Handle & data)
//! Insert pow task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * nntile_starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->nelems = nelems;
    args->alpha = alpha;
    args->exp = exp;
    // Submit task
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_RW, &data } }, args, sizeof(*args));
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

} // namespace nntile::haul
