/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/mask_scalar.cc
 * StarPU wrappers for mask_scalar operation
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/mask_scalar.hh"

// Standard libraries
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/mask_scalar.hh"

namespace nntile::haul
{

//! Constructor
template<typename T>
MaskScalar<std::tuple<T>>::MaskScalar():
    codelet("nntile_mask_scalar", &MaskScalar<std::tuple<T>>::cpu, nullptr, &MaskScalar<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! Mask scalar operation for StarPU buffer on CPU
template<typename T>
void MaskScalar<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    T *data = ::nntile::haul::buf_as<T>(buffers, 0);
    const bool_t* mask = ::nntile::haul::buf_as<bool_t>(buffers, 1);
    // Launch kernel
    kernel::mask_scalar::cpu<T>(args->nrows, args->ncols, mask, args->val,
            data);
}
// Specializations of CPU wrapper for accelerated types
template<>
void MaskScalar<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    MaskScalar<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void MaskScalar<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    MaskScalar<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void MaskScalar<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    MaskScalar<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}


//! Footprint for mask_scalar tasks
template<typename T>
std::uint64_t MaskScalar<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(&args->nrows, sizeof(args->nrows), hash);
    hash = ::nntile::haul::fnv1a(&args->ncols, sizeof(args->ncols), hash);
    return hash;
}

template<typename T>
void MaskScalar<std::tuple<T>>::submit(int starpu_worker_hint,
        Index nrows, Index ncols, ::nnhaul::Handle & mask, Scalar val, ::nnhaul::Handle & data)
//! Insert mask_scalar task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * nntile_starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->nrows = nrows;
    args->ncols = ncols;
    args->val = val;
    // Indicate maximal possible amount of writes as flops count
    double nflops = sizeof(T) * nrows * (ncols+1);
    // Submit task
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_RW, &data }, { STARPU_R, &mask } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class MaskScalar<std::tuple<nntile::fp64_t>>;
template class MaskScalar<std::tuple<nntile::fp32_t>>;
template class MaskScalar<std::tuple<nntile::fp32_fast_tf32_t>>;
template class MaskScalar<std::tuple<nntile::fp32_fast_fp16_t>>;
template class MaskScalar<std::tuple<nntile::fp32_fast_bf16_t>>;
template class MaskScalar<std::tuple<nntile::bf16_t>>;
template class MaskScalar<std::tuple<nntile::fp16_t>>;

//! Pack of mask_scalar operations for different types
mask_scalar_pack_t mask_scalar;

} // namespace nntile::haul
