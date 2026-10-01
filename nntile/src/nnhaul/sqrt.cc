/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/sqrt.cc
 * Sqrt operation on a StarPU buffer
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/sqrt.hh"

// Standard libraries
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/sqrt.hh"

namespace nntile::haul
{

//! Constructor
template<typename T>
Sqrt<std::tuple<T>>::Sqrt():
    codelet("nntile_sqrt", &Sqrt<std::tuple<T>>::cpu, nullptr, &Sqrt<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! Apply sqrt to StarPU buffer on CPU
template<typename T>
void Sqrt<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    const T *src = ::nntile::haul::buf_as<T>(buffers, 0);
    T *dst = ::nntile::haul::buf_as<T>(buffers, 1);
    // Launch kernel
    kernel::sqrt::cpu<T>(args->nelems, src, dst);
}

// Specializations of CPU wrapper for accelerated types
template<>
void Sqrt<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Sqrt<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Sqrt<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Sqrt<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Sqrt<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Sqrt<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}


//! Footprint for sqrt tasks that depends only on nelems
template<typename T>
std::uint64_t Sqrt<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(&args->nelems, sizeof(args->nelems), hash);
    return hash;
}

template<typename T>
void Sqrt<std::tuple<T>>::submit(int starpu_worker_hint, Index nelems, ::nnhaul::Handle & src, ::nnhaul::Handle & dst)
{
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->nelems = nelems;
    //double nflops = 5 * nelems;
    // Submit task
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &src }, { STARPU_W, &dst } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class Sqrt<std::tuple<nntile::fp64_t>>;
template class Sqrt<std::tuple<nntile::fp32_t>>;
template class Sqrt<std::tuple<nntile::fp32_fast_tf32_t>>;
template class Sqrt<std::tuple<nntile::fp32_fast_fp16_t>>;
template class Sqrt<std::tuple<nntile::fp32_fast_bf16_t>>;
template class Sqrt<std::tuple<nntile::bf16_t>>;

//! Pack of sqrt operations for different types
sqrt_pack_t sqrt;

} // namespace nntile::haul
