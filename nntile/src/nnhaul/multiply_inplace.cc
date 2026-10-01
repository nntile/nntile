/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/multiply_inplace.cc
 * Per-element multiplication of two StarPU buffers
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/multiply_inplace.hh"

// Standard libraries
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/multiply_inplace.hh"

namespace nntile::haul
{

//! Constructor
template<typename T>
MultiplyInplace<std::tuple<T>>::MultiplyInplace():
    codelet("nntile_multiply_inplace", &MultiplyInplace<std::tuple<T>>::cpu, nullptr, &MultiplyInplace<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! Apply multiply on StarPU buffers on CPU
template<typename T>
void MultiplyInplace<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    Index nelems = args->nelems;
    Scalar alpha = args->alpha;
    // Get interfaces
    const T *src = ::nntile::haul::buf_as<T>(buffers, 0);
    T *dst = ::nntile::haul::buf_as<T>(buffers, 1);
    // Launch kernel
    kernel::multiply_inplace::cpu<T>(nelems, alpha, src, dst);
}

// Specializations of CPU wrapper for accelerated types
template<>
void MultiplyInplace<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    MultiplyInplace<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void MultiplyInplace<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    MultiplyInplace<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void MultiplyInplace<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    MultiplyInplace<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}


//! Footprint for multiply_inplace operation
template<typename T>
std::uint64_t MultiplyInplace<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(&args->nelems, sizeof(args->nelems), hash);
    hash = ::nntile::haul::fnv1a(&args->alpha, sizeof(args->alpha), hash);
    return hash;
}

//! Submit multiply_inplace operation
template<typename T>
void MultiplyInplace<std::tuple<T>>::submit(int starpu_worker_hint, Index nelems, Scalar alpha, ::nnhaul::Handle & src, ::nnhaul::Handle & dst)
{
    // Codelet arguments
    args_t *args = (args_t*)std::malloc(sizeof(*args));
    args->nelems = nelems;
    args->alpha = alpha;
    // Put amount of read-write bytes into flop count
    double nflops = sizeof(T) * 3 * nelems;
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &src }, { STARPU_RW, &dst } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class MultiplyInplace<std::tuple<nntile::fp64_t>>;
template class MultiplyInplace<std::tuple<nntile::fp32_t>>;
template class MultiplyInplace<std::tuple<nntile::fp32_fast_tf32_t>>;
template class MultiplyInplace<std::tuple<nntile::fp32_fast_fp16_t>>;
template class MultiplyInplace<std::tuple<nntile::fp32_fast_bf16_t>>;
template class MultiplyInplace<std::tuple<nntile::bf16_t>>;
template class MultiplyInplace<std::tuple<nntile::fp16_t>>;

//! Pack of multiply_inplace operations for different types
multiply_inplace_pack_t multiply_inplace;

} // namespace nntile::haul
