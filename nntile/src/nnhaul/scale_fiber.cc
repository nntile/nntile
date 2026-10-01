/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/scale_fiber.cc
 * StarPU wrappers for scaling of a tensor with a broadcasted fiber
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/scale_fiber.hh"

// Standard libraries
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/scale_fiber.hh"
#include "nntile/nnhaul/ops/clear.hh"

namespace nntile::haul
{

//! Constructor
template<typename T>
ScaleFiber<std::tuple<T>>::ScaleFiber():
    codelet("nntile_scale_fiber", &ScaleFiber<std::tuple<T>>::cpu, nullptr, &ScaleFiber<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! Apply scale_fiber operation on StarPU buffers on CPU
template<typename T>
void ScaleFiber<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    const T *src = ::nntile::haul::buf_as<T>(buffers, 0);
    T *dst = ::nntile::haul::buf_as<T>(buffers, 1);
    // Launch kernel
    kernel::scale_fiber::cpu<T>(
        args->m,
        args->n,
        args->k,
        args->batch,
        args->alpha,
        src,
        dst
    );
}

// Specializations of CPU wrapper for accelerated types
template<>
void ScaleFiber<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    ScaleFiber<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void ScaleFiber<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    ScaleFiber<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void ScaleFiber<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    ScaleFiber<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}


//! Footprint for scale_fiber tasks
template<typename T>
std::uint64_t ScaleFiber<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(&args->m, sizeof(args->m), hash);
    hash = ::nntile::haul::fnv1a(&args->n, sizeof(args->n), hash);
    hash = ::nntile::haul::fnv1a(&args->k, sizeof(args->k), hash);
    hash = ::nntile::haul::fnv1a(&args->batch, sizeof(args->batch), hash);
    return hash;
}

//! Submit scale_fiber task
template<typename T>
void ScaleFiber<std::tuple<T>>::submit(int starpu_worker_hint,
    Index m,
    Index n,
    Index k,
    Index batch,
    Scalar alpha,
    ::nnhaul::Handle & src,
    ::nnhaul::Handle & dst
)
{
    // Reduce to clear buffer if alpha is zero
    if(alpha == 0.0)
    {
        clear.submit(starpu_worker_hint, dst);
        return;
    }
    // Codelet arguments
    args_t* args = (args_t*)std::malloc(sizeof(*args));
    args->m = m;
    args->n = n;
    args->k = k;
    args->batch = batch;
    args->alpha = alpha;
    // Put amount of bytes read and write inplace of gflops
    double nflops = sizeof(T) * batch * k * m * n;
    // Submit task
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &src }, { STARPU_W, &dst } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class ScaleFiber<std::tuple<nntile::fp64_t>>;
template class ScaleFiber<std::tuple<nntile::fp32_t>>;
template class ScaleFiber<std::tuple<nntile::fp32_fast_tf32_t>>;
template class ScaleFiber<std::tuple<nntile::fp32_fast_fp16_t>>;
template class ScaleFiber<std::tuple<nntile::fp32_fast_bf16_t>>;
template class ScaleFiber<std::tuple<nntile::bf16_t>>;
template class ScaleFiber<std::tuple<nntile::fp16_t>>;

//! Pack of scale_fiber operations for different types
scale_fiber_pack_t scale_fiber;

} // namespace nntile::haul
