/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/scale_slice.cc
 * StarPU wrappers for scaling of a broadcasted slice
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/scale_slice.hh"

// Standard libraries
#include <cstdint>
#include <cstdlib>

// Other NNTile headers
#include "nntile/kernel/scale_slice.hh"

namespace nntile::haul
{

//! Constructor
template<typename T>
ScaleSlice<std::tuple<T>>::ScaleSlice():
    codelet("nntile_scale_slice", &ScaleSlice<std::tuple<T>>::cpu, nullptr, &ScaleSlice<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! StarPU wrapper for kernel::scale_slice::cpu<T>
template<typename T>
void ScaleSlice<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    const T *src = ::nntile::haul::buf_as<T>(buffers, 0);
    T *dst = ::nntile::haul::buf_as<T>(buffers, 1);
    // Launch kernel
    kernel::scale_slice::cpu<T>(
        args->m, args->n, args->k, args->alpha, src, dst);
}

// Specializations of CPU wrapper for accelerated types
template<>
void ScaleSlice<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    ScaleSlice<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void ScaleSlice<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    ScaleSlice<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void ScaleSlice<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    ScaleSlice<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}


//! Footprint for scale_slice tasks
template<typename T>
std::uint64_t ScaleSlice<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(&args->m, sizeof(args->m), hash);
    hash = ::nntile::haul::fnv1a(&args->n, sizeof(args->n), hash);
    hash = ::nntile::haul::fnv1a(&args->k, sizeof(args->k), hash);
    return hash;
}

template<typename T>
void ScaleSlice<std::tuple<T>>::submit(int starpu_worker_hint,
    Index m,
    Index n,
    Index k,
    Scalar alpha,
    ::nnhaul::Handle & src,
    ::nnhaul::Handle & dst
)
//! Insert scale_slice task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * nntile_starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    // Codelet arguments
    args_t *args = (args_t*)std::malloc(sizeof(*args));
    args->m = m;
    args->n = n;
    args->k = k;
    args->alpha = alpha;
    // Put amount of bytes read and write inplace of gflops
    double nflops = sizeof(T) * m * (k+1) * n;
    // Submit task
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &src }, { STARPU_W, &dst } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class ScaleSlice<std::tuple<nntile::fp64_t>>;
template class ScaleSlice<std::tuple<nntile::fp32_t>>;
template class ScaleSlice<std::tuple<nntile::fp32_fast_tf32_t>>;
template class ScaleSlice<std::tuple<nntile::fp32_fast_fp16_t>>;
template class ScaleSlice<std::tuple<nntile::fp32_fast_bf16_t>>;
template class ScaleSlice<std::tuple<nntile::bf16_t>>;
template class ScaleSlice<std::tuple<nntile::fp16_t>>;

//! Pack of scale_slice operations for different types
scale_slice_pack_t scale_slice;

} // namespace nntile::haul
