/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/gelutanh_backward.cc
 * Backward approximate GeLU operation on a StarPU buffer
 *
 * @version 1.1.0
 * */

// Corresponding headers
#include "nntile/nnhaul/ops/gelutanh_backward.hh"

// Standard libraries
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/gelutanh_backward.hh"

namespace nntile::haul
{

//! Constructor
template<typename T>
GeluTanhBackward<std::tuple<T>>::GeluTanhBackward():
    codelet("nntile_gelutanh_backward", &GeluTanhBackward<std::tuple<T>>::cpu, nullptr, &GeluTanhBackward<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

template<typename T>
void GeluTanhBackward<std::tuple<T>>::cpu(void *buffers[], void *cl_args) noexcept
{
    // Get arguments
    args_t const *args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    const T *x = ::nntile::haul::buf_as<T>(buffers, 0);
    const T *dy = ::nntile::haul::buf_as<T>(buffers, 1);
    T *dx = ::nntile::haul::buf_as<T>(buffers, 2);
    // Launch kernel
    kernel::gelutanh_backward::cpu<T>(args->nelems, args->alpha, x, dy, args->beta, dx);
}

// Specializations of CPU wrapper for accelerated types
template<>
void GeluTanhBackward<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    GeluTanhBackward<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void GeluTanhBackward<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    GeluTanhBackward<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void GeluTanhBackward<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    GeluTanhBackward<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}


//! Footprint for add tasks that depends only on cl_arg
template<typename T>
std::uint64_t GeluTanhBackward<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(&args->nelems, sizeof(args->nelems), hash);
    hash = ::nntile::haul::fnv1a(&args->alpha, sizeof(args->alpha), hash);
    hash = ::nntile::haul::fnv1a(&args->beta, sizeof(args->beta), hash);
    return hash;
}

template<typename T>
void GeluTanhBackward<std::tuple<T>>::submit(int starpu_worker_hint, Index nelems, Scalar alpha,
        ::nnhaul::Handle & x, ::nnhaul::Handle & dy, Scalar beta, ::nnhaul::Handle & dx)
//! Insert gelutanh_backward task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * nntile_starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    starpu_data_access_mode dx_mode;
    if(beta == 0.0)
    {
        dx_mode = STARPU_W;
    }
    else
    {
        dx_mode = STARPU_RW;
    }
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->nelems = nelems;
    args->alpha = alpha;
    args->beta = beta;
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &x }, { STARPU_R, &dy }, { dx_mode, &dx } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class GeluTanhBackward<std::tuple<nntile::fp64_t>>;
template class GeluTanhBackward<std::tuple<nntile::fp32_t>>;
template class GeluTanhBackward<std::tuple<nntile::fp32_fast_tf32_t>>;
template class GeluTanhBackward<std::tuple<nntile::fp32_fast_fp16_t>>;
template class GeluTanhBackward<std::tuple<nntile::fp32_fast_bf16_t>>;
template class GeluTanhBackward<std::tuple<nntile::bf16_t>>;
template class GeluTanhBackward<std::tuple<nntile::fp16_t>>;

//! Pack of gelutanh_backward operations for different types
gelutanh_backward_pack_t gelutanh_backward;

} // namespace nntile::haul
