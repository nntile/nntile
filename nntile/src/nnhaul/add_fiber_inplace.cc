/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/add_fiber_inplace.cc
 * StarPU wrappers for addition of a tensor and a broadcasted fiber
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/add_fiber_inplace.hh"

// Standard libraries
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/add_fiber_inplace.hh"
#include "nntile/nnhaul/ops/scale_inplace.hh"
#include "nntile/nnhaul/ops/scale_fiber.hh"

namespace nntile::haul
{

//! Constructor
template<typename T>
AddFiberInplace<std::tuple<T>>::AddFiberInplace():
    codelet(
        "nntile_add_fiber_inplace",
        &AddFiberInplace<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &AddFiberInplace<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &AddFiberInplace<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! Apply add_fiber_inplace operation on StarPU buffers on CPU
template<typename T>
void AddFiberInplace<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    const T *src = ::nntile::haul::buf_as<T>(buffers, 0);
    T *dst = ::nntile::haul::buf_as<T>(buffers, 1);
    // Launch kernel
    kernel::add_fiber_inplace::cpu<T>(
        args->m,
        args->n,
        args->k,
        args->batch,
        args->alpha,
        src,
        args->beta,
        dst
    );
}

// Specializations of CPU wrapper for accelerated types
template<>
void AddFiberInplace<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AddFiberInplace<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void AddFiberInplace<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AddFiberInplace<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void AddFiberInplace<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AddFiberInplace<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! Apply add_fiber_inplace operation on StarPU buffer on CUDA
template<typename T>
void AddFiberInplace<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t*>(cl_args);
    // Get interfaces
    const T *src = ::nntile::haul::buf_as<T>(buffers, 0);
    T *dst = ::nntile::haul::buf_as<T>(buffers, 1);
    // Get CUDA stream
    cudaStream_t stream = ::nnhaul::cuda_stream();
    // Launch kernel
    kernel::add_fiber_inplace::cuda<T>(
        stream,
        args->m,
        args->n,
        args->k,
        args->batch,
        args->alpha,
        src,
        args->beta,
        dst
    );
}

// Specializations of CUDA wrapper for accelerated types
template<>
void AddFiberInplace<std::tuple<fp32_fast_tf32_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    AddFiberInplace<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void AddFiberInplace<std::tuple<fp32_fast_fp16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    AddFiberInplace<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void AddFiberInplace<std::tuple<fp32_fast_bf16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    AddFiberInplace<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for add_fiber_inplace tasks
template<typename T>
std::uint64_t AddFiberInplace<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(&args->m, sizeof(args->m), hash);
    hash = ::nntile::haul::fnv1a(&args->n, sizeof(args->n), hash);
    hash = ::nntile::haul::fnv1a(&args->k, sizeof(args->k), hash);
    hash = ::nntile::haul::fnv1a(&args->batch, sizeof(args->batch), hash);
    return hash;
}

//! Submit add_fiber_inplace task
template<typename T>
void AddFiberInplace<std::tuple<T>>::submit(int starpu_worker_hint,
    Index m,
    Index n,
    Index k,
    Index batch,
    Scalar alpha,
    ::nnhaul::Handle & src,
    Scalar beta,
    ::nnhaul::Handle & dst
)
{
    // If alpha is zero, then this operation reduces to scale_inplace
    if(alpha == 0.0)
    {
        scale_inplace.submit<std::tuple<T>>(starpu_worker_hint, m*n*k*batch, beta, dst);
        return;
    }
    // If beta is zero, then this operation reduces to scale_fiber
    if(beta == 0.0)
    {
        scale_fiber.submit<std::tuple<T>>(starpu_worker_hint, m, n, k, batch, alpha, src, dst);
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
    args_t* args = (args_t*)std::malloc(sizeof(*args));
    args->m = m;
    args->n = n;
    args->k = k;
    args->batch = batch;
    args->alpha = alpha;
    args->beta = beta;
    // Put amount of bytes read and write inplace of gflops
    double nflops = sizeof(T) * m * (2*k+1) * n * batch;
    // Submit task
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &src }, { dst_mode, &dst } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class AddFiberInplace<std::tuple<nntile::fp64_t>>;
template class AddFiberInplace<std::tuple<nntile::fp32_t>>;
template class AddFiberInplace<std::tuple<nntile::fp32_fast_tf32_t>>;
template class AddFiberInplace<std::tuple<nntile::fp32_fast_fp16_t>>;
template class AddFiberInplace<std::tuple<nntile::fp32_fast_bf16_t>>;
template class AddFiberInplace<std::tuple<nntile::bf16_t>>;
template class AddFiberInplace<std::tuple<nntile::fp16_t>>;

//! Pack of add_fiber_inplace operations for different types
add_fiber_inplace_pack_t add_fiber_inplace;

} // namespace nntile::haul
