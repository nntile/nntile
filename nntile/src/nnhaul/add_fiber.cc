/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/add_fiber.cc
 * StarPU wrappers for addition of a tensor and a broadcasted fiber
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/add_fiber.hh"

// Standard libraries
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/add_fiber.hh"
#include "nntile/nnhaul/ops/scale.hh"
#include "nntile/nnhaul/ops/scale_fiber.hh"

namespace nntile::haul
{

//! Constructor
template<typename T>
AddFiber<std::tuple<T>>::AddFiber():
    codelet("nntile_add_fiber", &AddFiber<std::tuple<T>>::cpu, nullptr, &AddFiber<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! Apply add_fiber operation on StarPU buffers on CPU
template<typename T>
void AddFiber<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    const T *src1 = ::nntile::haul::buf_as<T>(buffers, 0);
    const T *src2 = ::nntile::haul::buf_as<T>(buffers, 1);
    T *dst = ::nntile::haul::buf_as<T>(buffers, 2);
    // Launch kernel
    kernel::add_fiber::cpu<T>(
        args->m,
        args->n,
        args->k,
        args->batch,
        args->alpha,
        src1,
        args->beta,
        src2,
        dst
    );
}

// Specializations of CPU wrapper for accelerated types
template<>
void AddFiber<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AddFiber<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void AddFiber<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AddFiber<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void AddFiber<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AddFiber<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}


//! Footprint for add_fiber tasks
template<typename T>
std::uint64_t AddFiber<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
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

//! Submit add_fiber task
template<typename T>
void AddFiber<std::tuple<T>>::submit(int starpu_worker_hint,
    Index m,
    Index n,
    Index k,
    Index batch,
    Scalar alpha,
    ::nnhaul::Handle & src1,
    Scalar beta,
    ::nnhaul::Handle & src2,
    ::nnhaul::Handle & dst
)
{
    // If alpha is zero, then this operation reduces to scale
    if(alpha == 0.0)
    {
        scale.submit<std::tuple<T>>(starpu_worker_hint, m*n*k*batch, beta, src2, dst);
        return;
    }
    // If beta is zero, then this operation reduces to scale_fiber
    if(beta == 0.0)
    {
        scale_fiber.submit<std::tuple<T>>(starpu_worker_hint, m, n, k, batch, alpha, src1, dst);
        return;
    }
    // Codelet arguments
    args_t* args = (args_t*)std::malloc(sizeof(*args));
    args->m = m;
    args->n = n;
    args->k = k;
    args->batch = batch;
    args->alpha = alpha;
    args->beta = beta;
    double nflops = batch * k * (2*m*n+1);
    // Submit task
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &src1 }, { STARPU_R, &src2 }, { STARPU_W, &dst } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class AddFiber<std::tuple<nntile::fp64_t>>;
template class AddFiber<std::tuple<nntile::fp32_t>>;
template class AddFiber<std::tuple<nntile::fp32_fast_tf32_t>>;
template class AddFiber<std::tuple<nntile::fp32_fast_fp16_t>>;
template class AddFiber<std::tuple<nntile::fp32_fast_bf16_t>>;
template class AddFiber<std::tuple<nntile::bf16_t>>;

//! Pack of add_fiber operations for different types
add_fiber_pack_t add_fiber;

} // namespace nntile::haul
