/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/add_inplace.cc
 * Add operation on a StarPU buffers
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/add_inplace.hh"

// Standard libraries
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/add_inplace.hh"
#include "nntile/nnhaul/ops/scale.hh"
#include "nntile/nnhaul/ops/scale_inplace.hh"

//! StarPU wrappers for add_inplace operation
namespace nntile::haul
{

//! Constructor
template<typename T>
AddInplace<std::tuple<T>>::AddInplace():
    codelet("nntile_add_inplace", &AddInplace<std::tuple<T>>::cpu, nullptr, &AddInplace<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! Apply add operation for StarPU buffers in CPU
template<typename T>
void AddInplace<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    const T *src = ::nntile::haul::buf_as<T>(buffers, 0);
    T *dst = ::nntile::haul::buf_as<T>(buffers, 1);
    // Launch kernel
    kernel::add_inplace::cpu<T>(
        args->nelems, args->alpha, src, args->beta, dst);
}

// Specializations of CPU wrapper for accelerated types
template<>
void AddInplace<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AddInplace<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void AddInplace<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AddInplace<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void AddInplace<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AddInplace<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}


//! Footprint for add tasks that depends only on cl_arg
template<typename T>
std::uint64_t AddInplace<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(&args->nelems, sizeof(args->nelems), hash);
    return hash;
}

//! Submit add_inplace task
template<typename T>
void AddInplace<std::tuple<T>>::submit(int starpu_worker_hint,
    Index nelems,
    Scalar alpha,
    ::nnhaul::Handle & src,
    Scalar beta,
    ::nnhaul::Handle & dst
)
{
    // If alpha is zero then reduce to scale_inplace
    if(alpha == 0.0)
    {
        scale_inplace.submit<std::tuple<T>>(starpu_worker_hint, nelems, beta, dst);
        return;
    }
    // If beta is zero this function reduces to scale
    if(beta == 0.0)
    {
        scale.submit<std::tuple<T>>(starpu_worker_hint, nelems, alpha, src, dst);
        return;
    }
    // Access mode for the dst handle
    enum starpu_data_access_mode dst_mode;
    if(beta == 1.0)
    {
        dst_mode = static_cast<starpu_data_access_mode>(STARPU_RW | STARPU_COMMUTE);
    }
    else
    {
        dst_mode = STARPU_RW;
    }
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->nelems = nelems;
    args->alpha = alpha;
    args->beta = beta;
    double nflops = 2 * nelems;
    // Submit task
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &src }, { dst_mode, &dst } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class AddInplace<std::tuple<nntile::fp64_t>>;
template class AddInplace<std::tuple<nntile::fp32_t>>;
template class AddInplace<std::tuple<nntile::fp32_fast_tf32_t>>;
template class AddInplace<std::tuple<nntile::fp32_fast_fp16_t>>;
template class AddInplace<std::tuple<nntile::fp32_fast_bf16_t>>;
template class AddInplace<std::tuple<nntile::bf16_t>>;
template class AddInplace<std::tuple<nntile::fp16_t>>;

//! Pack of add_inplace operations for different types
add_inplace_pack_t add_inplace;

} // namespace nntile::haul
