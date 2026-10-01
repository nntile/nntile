/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/multiply.cc
 * Per-element product of two StarPU buffers
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/multiply.hh"

// Standard libraries
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/multiply.hh"

namespace nntile::haul
{

//! Constructor
template<typename T>
Multiply<std::tuple<T>>::Multiply():
    codelet("nntile_multiply", &Multiply<std::tuple<T>>::cpu, nullptr, &Multiply<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! StarPU wrapper for kernel::multiply::cpu<T>
template<typename T>
void Multiply<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    Index nelems = args->nelems;
    Scalar alpha = args->alpha;
    // Get interfaces
    const T *src1 = ::nntile::haul::buf_as<T>(buffers, 0);
    const T *src2 = ::nntile::haul::buf_as<T>(buffers, 1);
    T *dst = ::nntile::haul::buf_as<T>(buffers, 2);
    // Launch kernel
    kernel::multiply::cpu<T>(nelems, alpha, src1, src2, dst);
}

// Specializations of CPU wrapper for accelerated types
template<>
void Multiply<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Multiply<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Multiply<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Multiply<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Multiply<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Multiply<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}


//! Footprint for add tasks that depends only on cl_arg
template<typename T>
std::uint64_t Multiply<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(&args->nelems, sizeof(args->nelems), hash);
    hash = ::nntile::haul::fnv1a(&args->alpha, sizeof(args->alpha), hash);
    return hash;
}

template<typename T>
void Multiply<std::tuple<T>>::submit(int starpu_worker_hint, Index nelems, Scalar alpha, ::nnhaul::Handle & src1, ::nnhaul::Handle & src2, ::nnhaul::Handle & dst)
{
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->nelems = nelems;
    args->alpha = alpha;
    // Put amount of read-write bytes into flop count
    double nflops = sizeof(T) * 3 * nelems;
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &src1 }, { STARPU_R, &src2 }, { STARPU_W, &dst } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class Multiply<std::tuple<nntile::fp64_t>>;
template class Multiply<std::tuple<nntile::fp32_t>>;
template class Multiply<std::tuple<nntile::fp32_fast_tf32_t>>;
template class Multiply<std::tuple<nntile::fp32_fast_fp16_t>>;
template class Multiply<std::tuple<nntile::fp32_fast_bf16_t>>;
template class Multiply<std::tuple<nntile::bf16_t>>;
template class Multiply<std::tuple<nntile::fp16_t>>;

//! Pack of multiply operations for different types
multiply_pack_t multiply;

} // namespace nntile::haul
