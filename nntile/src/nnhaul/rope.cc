/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/rope.cc
 * Rotary positional embedding
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/rope.hh"

// Standard libraries
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/rope.hh"

namespace nntile::haul
{

//! Constructor
template<typename T>
Rope<std::tuple<T>>::Rope():
    codelet("nntile_rope", &Rope<std::tuple<T>>::cpu, nullptr, &Rope<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! StarPU wrapper for kernel::rope::cpu<T>
template<typename T>
void Rope<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    const T *sin = ::nntile::haul::buf_as<T>(buffers, 0);
    const T *cos = ::nntile::haul::buf_as<T>(buffers, 1);
    const T *src = ::nntile::haul::buf_as<T>(buffers, 2);
    T *dst = ::nntile::haul::buf_as<T>(buffers, 3);
    // Launch kernel
    kernel::rope::cpu<T>(args->m, args->n, sin, cos, src, dst);
}

// Specializations of CPU wrapper for accelerated types
template<>
void Rope<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Rope<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Rope<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Rope<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Rope<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Rope<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}


//! Footprint for rope tasks
template<typename T>
std::uint64_t Rope<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Apply hash over parameters m, and k
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(&args->m, sizeof(args->m), hash);
    hash = ::nntile::haul::fnv1a(&args->n, sizeof(args->n), hash);
    return hash;
}

template<typename T>
void Rope<std::tuple<T>>::submit(int starpu_worker_hint, Index m, Index n, ::nnhaul::Handle & sin, ::nnhaul::Handle & cos, ::nnhaul::Handle & src, ::nnhaul::Handle & dst)
//! Insert rope task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * nntile_starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->m = m;
    args->n = n;
    // Submit task
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &sin }, { STARPU_R, &cos }, { STARPU_R, &src }, { STARPU_W, &dst } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class Rope<std::tuple<nntile::fp64_t>>;
template class Rope<std::tuple<nntile::fp32_t>>;
template class Rope<std::tuple<nntile::fp32_fast_tf32_t>>;
template class Rope<std::tuple<nntile::fp32_fast_fp16_t>>;
template class Rope<std::tuple<nntile::fp32_fast_bf16_t>>;
template class Rope<std::tuple<nntile::fp16_t>>;
template class Rope<std::tuple<nntile::bf16_t>>;

//! Pack of rope operations for different types
rope_pack_t rope;

} // namespace nntile::haul
