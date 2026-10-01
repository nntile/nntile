/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/rope_backward.cc
 * Backward of rotary positional embedding
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/rope_backward.hh"

// Standard libraries
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/rope_backward.hh"

namespace nntile::haul
{

//! Constructor
template<typename T>
RopeBackward<std::tuple<T>>::RopeBackward():
    codelet("nntile_rope_backward", &RopeBackward<std::tuple<T>>::cpu, nullptr, &RopeBackward<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! StarPU wrapper for kernel::rope_backward::cpu<T>
template<typename T>
void RopeBackward<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    const T *sin = ::nntile::haul::buf_as<T>(buffers, 0);
    const T *cos = ::nntile::haul::buf_as<T>(buffers, 1);
    const T *dy = ::nntile::haul::buf_as<T>(buffers, 2);
    T *dx = ::nntile::haul::buf_as<T>(buffers, 3);
    // Launch kernel
    kernel::rope_backward::cpu<T>(args->m, args->n, sin, cos, dy, dx);
}

// Specializations of CPU wrapper for accelerated types
template<>
void RopeBackward<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    RopeBackward<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void RopeBackward<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    RopeBackward<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void RopeBackward<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    RopeBackward<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}


//! Footprint for rope_backward tasks
template<typename T>
std::uint64_t RopeBackward<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
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
void RopeBackward<std::tuple<T>>::submit(int starpu_worker_hint, Index m, Index n, ::nnhaul::Handle & sin, ::nnhaul::Handle & cos, ::nnhaul::Handle & dy, ::nnhaul::Handle & dx)
//! Insert rope_backward task into StarPU pool of tasks
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
        { { STARPU_R, &sin }, { STARPU_R, &cos }, { STARPU_R, &dy }, { STARPU_W, &dx } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class RopeBackward<std::tuple<nntile::fp64_t>>;
template class RopeBackward<std::tuple<nntile::fp32_t>>;
template class RopeBackward<std::tuple<nntile::fp32_fast_tf32_t>>;
template class RopeBackward<std::tuple<nntile::fp32_fast_fp16_t>>;
template class RopeBackward<std::tuple<nntile::fp32_fast_bf16_t>>;
template class RopeBackward<std::tuple<nntile::fp16_t>>;
template class RopeBackward<std::tuple<nntile::bf16_t>>;

//! Pack of rope_backward operations for different types
rope_backward_pack_t rope_backward;

} // namespace nntile::haul
