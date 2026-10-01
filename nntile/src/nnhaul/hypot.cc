/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/hypot.cc
 * hypot operation on a StarPU buffers
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/hypot.hh"

// Standard libraries
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/hypot.hh"
#include "nntile/nnhaul/ops/scale.hh"
#include "nntile/nnhaul/ops/clear.hh"

namespace nntile::haul
{

//! Constructor
template<typename T>
Hypot<std::tuple<T>>::Hypot():
    codelet(
        "nntile_hypot",
        &Hypot<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &Hypot<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &Hypot<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! Apply hypot operation for StarPU buffers in CPU
template<typename T>
void Hypot<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    const T *src1 = ::nntile::haul::buf_as<T>(buffers, 0);
    const T *src2 = ::nntile::haul::buf_as<T>(buffers, 1);
    T *dst = ::nntile::haul::buf_as<T>(buffers, 2);
    // Launch kernel
    kernel::hypot::cpu<T>(args->nelems, args->alpha, src1, args->beta, src2, dst);
}

// Specializations of CPU wrapper for accelerated types
template<>
void Hypot<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Hypot<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Hypot<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Hypot<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Hypot<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Hypot<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! Apply hypot for StarPU buffers on CUDA
template<typename T>
void Hypot<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    const T *src1 = ::nntile::haul::buf_as<T>(buffers, 0);
    const T *src2 = ::nntile::haul::buf_as<T>(buffers, 1);
    T *dst = ::nntile::haul::buf_as<T>(buffers, 2);
    // Get CUDA stream
    cudaStream_t stream = ::nnhaul::cuda_stream();
    // Launch kernel
    kernel::hypot::cuda<T>(stream, args->nelems, args->alpha, src1,
            args->beta, src2, dst);
}

// Specializations of CUDA wrapper for accelerated types
template<>
void Hypot<std::tuple<fp32_fast_tf32_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Hypot<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void Hypot<std::tuple<fp32_fast_fp16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Hypot<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void Hypot<std::tuple<fp32_fast_bf16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Hypot<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for hypot tasks that depends only on cl_arg
template<typename T>
std::uint64_t Hypot<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(&args->nelems, sizeof(args->nelems), hash);
    return hash;
}

//! Submit hypot task
template<typename T>
void Hypot<std::tuple<T>>::submit(int starpu_worker_hint,
        Index nelems, Scalar alpha, ::nnhaul::Handle & src1, Scalar beta, ::nnhaul::Handle & src2, ::nnhaul::Handle & dst)
//! Insert hypot task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * nntile_starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->nelems = nelems;
    args->alpha = alpha;
    args->beta = beta;
    // Submit task
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &src1 }, { STARPU_R, &src2 }, { STARPU_W, &dst } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class Hypot<std::tuple<nntile::fp64_t>>;
template class Hypot<std::tuple<nntile::fp32_t>>;
template class Hypot<std::tuple<nntile::fp32_fast_tf32_t>>;
template class Hypot<std::tuple<nntile::fp32_fast_fp16_t>>;
template class Hypot<std::tuple<nntile::fp32_fast_bf16_t>>;
template class Hypot<std::tuple<nntile::bf16_t>>;
template class Hypot<std::tuple<nntile::fp16_t>>;

//! Pack of hypot operations for different types
hypot_pack_t hypot;

} // namespace nntile::haul
