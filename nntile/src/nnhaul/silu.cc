/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/silu.cc
 * SiLU operation on a StarPU buffer
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/silu.hh"

// Standard libraries
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/silu.hh"

namespace nntile::haul
{

//! Constructor
template<typename T>
Silu<std::tuple<T>>::Silu():
    codelet("nntile_silu", &Silu<std::tuple<T>>::cpu, nullptr, &Silu<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! StarPU wrapper for kernel::silu::cpu<T>
template<typename T>
void Silu<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    const T *src = ::nntile::haul::buf_as<T>(buffers, 0);
    T *dst = ::nntile::haul::buf_as<T>(buffers, 1);
    // Launch kernel
    kernel::silu::cpu<T>(args->nelems, src, dst);
}

// Specializations of CPU wrapper for accelerated types
template<>
void Silu<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Silu<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Silu<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Silu<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Silu<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Silu<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}


template<typename T>
std::uint64_t Silu<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    auto args = reinterpret_cast<args_t const *>(cl_args);
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(&args->nelems, sizeof(args->nelems), hash);
    return hash;
}

template<typename T>
void Silu<std::tuple<T>>::submit(int starpu_worker_hint, Index nelems, ::nnhaul::Handle & src, ::nnhaul::Handle & dst)
{
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->nelems = nelems;
    // Submit task
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &src }, { STARPU_W, &dst } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class Silu<std::tuple<nntile::fp64_t>>;
template class Silu<std::tuple<nntile::fp32_t>>;
template class Silu<std::tuple<nntile::fp32_fast_tf32_t>>;
template class Silu<std::tuple<nntile::fp32_fast_fp16_t>>;
template class Silu<std::tuple<nntile::fp32_fast_bf16_t>>;
template class Silu<std::tuple<nntile::bf16_t>>;
template class Silu<std::tuple<nntile::fp16_t>>;

//! Pack of silu operations for different types
silu_pack_t silu;

} // namespace nntile::haul
