/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/sumprod_fiber.cc
 * Sums over slices into a fiber of a product of two StarPU buffers
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/sumprod_fiber.hh"

// Standard libraries
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/sumprod_fiber.hh"

namespace nntile::haul
{

//! Constructor
template<typename T>
SumProdFiber<std::tuple<T>>::SumProdFiber():
    codelet("nntile_sumprod_fiber", &SumProdFiber<std::tuple<T>>::cpu, nullptr, &SumProdFiber<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! StarPU wrapper for kernel::sumprod_fiber::cpu<T>
template<typename T>
void SumProdFiber<std::tuple<T>>::cpu(void *buffers[], void *cl_args) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    const T *src1 = ::nntile::haul::buf_as<T>(buffers, 0);
    const T *src2 = ::nntile::haul::buf_as<T>(buffers, 1);
    T *dst = ::nntile::haul::buf_as<T>(buffers, 2);
    // Launch kernel
    kernel::sumprod_fiber::cpu<T>(args->m, args->n, args->k, args->alpha,
            src1, src2, args->beta, dst);
}

// Specializations of CPU wrapper for accelerated types
template<>
void SumProdFiber<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    SumProdFiber<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void SumProdFiber<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    SumProdFiber<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void SumProdFiber<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    SumProdFiber<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}


//! Footprint for sumprod_fiber tasks
template<typename T>
std::uint64_t SumProdFiber<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Hash over alpha
    uint32_t hash = args->alpha == Scalar{0} ? -1 : 0;
    // Apply hash over parameters m, n and k
    hash = ::nntile::haul::fnv1a(&args->m, sizeof(args->m), hash);
    hash = ::nntile::haul::fnv1a(&args->n, sizeof(args->n), hash);
    hash = ::nntile::haul::fnv1a(&args->k, sizeof(args->k), hash);
    return hash;
}

template<typename T>
void SumProdFiber<std::tuple<T>>::submit(int starpu_worker_hint, Index m, Index n, Index k, Scalar alpha, ::nnhaul::Handle & src1, ::nnhaul::Handle & src2,
        Scalar beta, ::nnhaul::Handle & dst, int redux)
//! Insert sumprod_fiber task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * nntile_starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    // Access mode for the dst handle
    constexpr Scalar zero = 0, one = 1;
    enum starpu_data_access_mode dst_mode;
    if(beta == zero)
    {
        dst_mode = STARPU_W;
    }
    else if(beta == one)
    {
        if(redux != 0)
        {
            dst_mode = STARPU_REDUX;
        }
        else
        {
            dst_mode = static_cast<starpu_data_access_mode>(STARPU_RW | STARPU_COMMUTE);
        }
    }
    else
    {
        dst_mode = STARPU_RW;
    }
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->m = m;
    args->n = n;
    args->k = k;
    args->alpha = alpha;
    args->beta = beta;
    // Put amount of bytes read and write inplace of gflops
    size_t src1_nbytes = sizeof(T) * m * k * n;
    size_t dst_nbytes = sizeof(T) * k;
    double nflops = beta == 0.0 ? 2*src1_nbytes + dst_nbytes :
        2 * (src1_nbytes+dst_nbytes);
    // Submit task
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &src1 }, { STARPU_R, &src2 }, { dst_mode, &dst } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class SumProdFiber<std::tuple<nntile::fp64_t>>;
template class SumProdFiber<std::tuple<nntile::fp32_t>>;
template class SumProdFiber<std::tuple<nntile::fp32_fast_tf32_t>>;
template class SumProdFiber<std::tuple<nntile::fp32_fast_fp16_t>>;
template class SumProdFiber<std::tuple<nntile::fp32_fast_bf16_t>>;
template class SumProdFiber<std::tuple<nntile::bf16_t>>;
template class SumProdFiber<std::tuple<nntile::fp16_t>>;

//! Pack of sumprod_fiber operations for different types
sumprod_fiber_pack_t sumprod_fiber;

} // namespace nntile::haul
