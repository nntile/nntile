/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/add.cc
 * Add operation on a StarPU buffers
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/add.hh"

// Standard libraries
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/add.hh"
#include "nntile/nnhaul/ops/scale.hh"

namespace nntile::haul
{

//! Constructor
template<typename T>
Add<std::tuple<T>>::Add():
    codelet(
        "nntile_add",
        &Add<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &Add<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &Add<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! Apply add operation for StarPU buffers in CPU
template<typename T>
void Add<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    const T *src1 = ::nntile::haul::buf_as<T>(buffers, 0);
    const T *src2 = ::nntile::haul::buf_as<T>(buffers, 1);
    T *dst = ::nntile::haul::buf_as<T>(buffers, 2);
    // Launch kernel
    kernel::add::cpu<T>(
        args->nelems, args->alpha, src1, args->beta, src2, dst);
}

// Specializations of CPU wrapper for accelerated types
template<>
void Add<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Add<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Add<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Add<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Add<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Add<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! Apply add for NNHaul buffers on CUDA
template<typename T>
void Add<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    auto args = reinterpret_cast<args_t const *>(cl_args);
    const T *src1 = ::nntile::haul::buf_as<T>(buffers, 0);
    const T *src2 = ::nntile::haul::buf_as<T>(buffers, 1);
    T *dst = ::nntile::haul::buf_as<T>(buffers, 2);
    cudaStream_t stream = ::nnhaul::cuda_stream();
    std::fprintf(stderr,
        "nntile_add nelems=%lld src1=%p src2=%p dst=%p\n",
        static_cast<long long>(args->nelems),
        static_cast<void const *>(src1),
        static_cast<void const *>(src2),
        static_cast<void *>(dst));
    std::fflush(stderr);
    kernel::add::cuda<T>(
        stream,
        args->nelems,
        args->alpha,
        src1,
        args->beta,
        src2,
        dst);
}

template<>
void Add<std::tuple<fp32_fast_tf32_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    Add<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void Add<std::tuple<fp32_fast_fp16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    Add<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void Add<std::tuple<fp32_fast_bf16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    Add<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for add tasks that depends only on cl_arg
template<typename T>
std::uint64_t Add<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(&args->nelems, sizeof(args->nelems), hash);
    return hash;
}

template<typename T>
void Add<std::tuple<T>>::submit(int starpu_worker_hint,
    Index nelems,
    Scalar alpha,
    ::nnhaul::Handle & src1,
    Scalar beta,
    ::nnhaul::Handle & src2,
    ::nnhaul::Handle & dst
)
{
    constexpr Scalar zero = 0;
    // If beta is zero this function reduces to scale
    if(beta == zero)
    {
        // dst = alpha*src1
        scale.submit<std::tuple<T>>(starpu_worker_hint, nelems, alpha, src1, dst);
        return;
    }
    // If beta is non-zero and alpha is zero then reduce to scale
    if(alpha == zero)
    {
        // dst = beta*src2
        scale.submit<std::tuple<T>>(starpu_worker_hint, nelems, beta, src2, dst);
        return;
    }
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->nelems = nelems;
    args->alpha = alpha;
    args->beta = beta;
    // Put amount of bytes read and write inplace of gflops
    double nflops = sizeof(T) * 3 * nelems;
    // Submit task
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &src1 }, { STARPU_R, &src2 }, { STARPU_W, &dst } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class Add<std::tuple<nntile::fp64_t>>;
template class Add<std::tuple<nntile::fp32_t>>;
template class Add<std::tuple<nntile::fp32_fast_tf32_t>>;
template class Add<std::tuple<nntile::fp32_fast_fp16_t>>;
template class Add<std::tuple<nntile::fp32_fast_bf16_t>>;
template class Add<std::tuple<nntile::bf16_t>>;
template class Add<std::tuple<nntile::fp16_t>>;

//! Pack of add operations for different types
add_pack_t add;

} // namespace nntile::haul
