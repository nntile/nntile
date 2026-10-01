/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/gelu_inplace.cc
 * GeLU inplace operation on a StarPU buffer
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/gelu_inplace.hh"

// Standard libraries
#include <cstdint>
#include <cstdlib>

// Other NNTile headers
#include "nntile/kernel/gelu_inplace.hh"

namespace nntile::haul
{

//! Constructor
template<typename T>
GeluInplace<std::tuple<T>>::GeluInplace():
    codelet("nntile_gelu_inplace", &GeluInplace<std::tuple<T>>::cpu, nullptr, &GeluInplace<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! Apply gelu_inplace on StarPU buffer on CPU
template<typename T>
void GeluInplace<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    T *data = ::nntile::haul::buf_as<T>(buffers, 0);
    // Launch kernel
    kernel::gelu_inplace::cpu<T>(args->nelems, data);
}

// Specializations of CPU wrapper for accelerated types
template<>
void GeluInplace<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    GeluInplace<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void GeluInplace<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    GeluInplace<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void GeluInplace<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    GeluInplace<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}


//! Footprint for add tasks that depends only on cl_arg
template<typename T>
std::uint64_t GeluInplace<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(&args->nelems, sizeof(args->nelems), hash);
    return hash;
}

//! Submit gelu_inplace task
template<typename T>
void GeluInplace<std::tuple<T>>::submit(int starpu_worker_hint,
    Index nelems,
    ::nnhaul::Handle & data
)
{
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->nelems = nelems;
    // Submit task
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_RW, &data } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class GeluInplace<std::tuple<nntile::fp64_t>>;
template class GeluInplace<std::tuple<nntile::fp32_t>>;
template class GeluInplace<std::tuple<nntile::fp32_fast_tf32_t>>;
template class GeluInplace<std::tuple<nntile::fp32_fast_fp16_t>>;
template class GeluInplace<std::tuple<nntile::fp32_fast_bf16_t>>;
template class GeluInplace<std::tuple<nntile::bf16_t>>;
template class GeluInplace<std::tuple<nntile::fp16_t>>;

//! Pack of gelu_inplace operations for different types
gelu_inplace_pack_t gelu_inplace;

} // namespace nntile::haul
