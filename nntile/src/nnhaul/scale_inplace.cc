/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/scale_inplace.cc
 * Scale inplace operation on a StarPU buffers
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/scale_inplace.hh"

// Standard libraries
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/scale_inplace.hh"
#include "nntile/nnhaul/ops/clear.hh"

namespace nntile::haul
{

//! Constructor
template<typename T>
ScaleInplace<std::tuple<T>>::ScaleInplace():
    codelet("nntile_scale_inplace", &ScaleInplace<std::tuple<T>>::cpu, nullptr, &ScaleInplace<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! StarPU wrapper for kernel::scale_inplace::cpu<T>
template<typename T>
void ScaleInplace<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    T *data = ::nntile::haul::buf_as<T>(buffers, 0);
    // Launch kernel
    kernel::scale_inplace::cpu<T>(args->nelems, args->alpha, data);
}

// Specializations of CPU wrapper for accelerated types
template<>
void ScaleInplace<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    ScaleInplace<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void ScaleInplace<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    ScaleInplace<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void ScaleInplace<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    ScaleInplace<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}


//! Footprint for scale_inplace tasks that depends only on cl_arg
template<typename T>
std::uint64_t ScaleInplace<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(&args->nelems, sizeof(args->nelems), hash);
    return hash;
}

template<typename T>
void ScaleInplace<std::tuple<T>>::submit(int starpu_worker_hint,
    Index nelems, Scalar alpha, ::nnhaul::Handle & data)
{
    constexpr Scalar zero = 0.0;
    // if alpha is zero, function reduces to clear
    if(alpha == zero)
    {
        clear.submit(starpu_worker_hint, data);
        return;
    }
    // if alpha is one, function reduces to no-op
    if(alpha == 1.0)
    {
        return;
    }
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->nelems = nelems;
    args->alpha = alpha;
    // Submit task
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_RW, &data } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class ScaleInplace<std::tuple<nntile::fp64_t>>;
template class ScaleInplace<std::tuple<nntile::fp32_t>>;
template class ScaleInplace<std::tuple<nntile::fp32_fast_tf32_t>>;
template class ScaleInplace<std::tuple<nntile::fp32_fast_fp16_t>>;
template class ScaleInplace<std::tuple<nntile::fp32_fast_bf16_t>>;
template class ScaleInplace<std::tuple<nntile::bf16_t>>;
template class ScaleInplace<std::tuple<nntile::fp16_t>>;

//! Pack of scale_inplace operations for different types
scale_inplace_pack_t scale_inplace;

} // namespace nntile::haul
