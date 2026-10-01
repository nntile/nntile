/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/gemm.cc
 * GEMM operation for StarPU buffers
 *
 * @version 1.1.0
 * */

// Corresponding headers
#include "nntile/nnhaul/ops/gemm.hh"

// Standard libraries
#include <cstdint>
#include <cstdlib>

// Third-party headers

// Other NNTile headers
#include "nntile/kernel/cblas.hh"
#include "nntile/kernel/cublas.hh"

namespace nntile::haul
{

//! Constructor
template<typename T>
Gemm<std::tuple<T>>::Gemm():
    codelet(
        "nntile_gemm",
        &Gemm<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &Gemm<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &Gemm<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default.
    // Unsupported types no-op inside the CPU kernel.
}

#ifdef NNTILE_USE_CBLAS // CPU implementation requires CBLAS
//! GEMM for contiguous matrices without padding through StarPU buffers
template<typename T>
void Gemm<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    // Launch kernel
    const T *A = ::nntile::haul::buf_as<T>(buffers, 0);
    const T *B = ::nntile::haul::buf_as<T>(buffers, 1);
    T *C = ::nntile::haul::buf_as<T>(buffers, 2);
    // Call corresponding CBLAS routine if supported, otherwise do nothing
    if constexpr (kernel::cblas::gemm_is_supported<T>)
    {
        kernel::cblas::gemm<T>(
            args->transA,
            args->transB,
            args->m,
            args->n,
            args->k,
            args->batch,
            args->alpha,
            A,
            B,
            args->beta,
            C
        );
    }
}
#endif // NNTILE_USE_CBLAS

#ifdef NNTILE_USE_CUDA // CUDA implementation requires cuBLAS
//! GEMM for contiguous matrices without padding through StarPU buffers
template<typename T>
void Gemm<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    // Launch kernel
    const T *A = ::nntile::haul::buf_as<T>(buffers, 0);
    const T *B = ::nntile::haul::buf_as<T>(buffers, 1);
    T *C = ::nntile::haul::buf_as<T>(buffers, 2);
    // Get cuBLAS handle and CUDA stream
    cublasHandle_t handle = ::nnhaul::cublas_handle();
    cudaStream_t stream = ::nnhaul::cuda_stream();
    cublasSetStream(handle, stream);
    // alpha and beta parameters of GEMM operation are on CPU host
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    // Call corresponding cuBLAS routine
    kernel::cublas::gemm<T>(
        handle,
        args->transA,
        args->transB,
        args->m,
        args->n,
        args->k,
        args->batch,
        args->alpha,
        A,
        B,
        args->beta,
        C
    );
}
#endif //NNTILE_USE_CUDA

//! Footprint for GEMM tasks that depends on transA, transB, M, N, K, batch and alpha
template<typename T>
std::uint64_t Gemm<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // In case alpha is zero, entire gemm is unnecessary so it is better to
    // give it a different footprint since gemm time will be totally different
    uint32_t hash = args->alpha == Scalar{0} ? -1 : 0;
    // Single codelet is used for all combinations of transA and transB
    hash = ::nntile::haul::fnv1a(&args->transA, sizeof(args->transA), hash);
    hash = ::nntile::haul::fnv1a(&args->transB, sizeof(args->transB), hash);
    // Apply hash over parameters M, N and K. This way if we swap values of M,
    // N and K total size of buffers will remain the same, but the footprint
    // will be different
    hash = ::nntile::haul::fnv1a(&args->m, sizeof(args->m), hash);
    hash = ::nntile::haul::fnv1a(&args->n, sizeof(args->n), hash);
    hash = ::nntile::haul::fnv1a(&args->k, sizeof(args->k), hash);
    hash = ::nntile::haul::fnv1a(&args->batch, sizeof(args->batch), hash);
    return hash;
}

template<typename T>
void Gemm<std::tuple<T>>::submit(int starpu_worker_hint, const TransOp &transA, const TransOp &transB, Index m, Index n,
        Index k, Index batch, Scalar alpha, ::nnhaul::Handle & A, ::nnhaul::Handle & B, Scalar beta,
        ::nnhaul::Handle & C, int redux)
{
    if(redux != 0)
    {
        throw std::runtime_error("redux != 0 is not supported");
    }
    // Check that matrix sizes fit proper types for underlying CBLAS
#ifdef NNTILE_USE_CBLAS
    if(static_cast<CBLAS_INT>(m) != m)
    {
        throw std::runtime_error("GEMM size M does not fit CBLAS_INT");
    }
    if(static_cast<CBLAS_INT>(n) != n)
    {
        throw std::runtime_error("GEMM size N does not fit CBLAS_INT");
    }
    if(static_cast<CBLAS_INT>(k) != k)
    {
        throw std::runtime_error("GEMM size K does not fit CBLAS_INT");
    }
#endif // NNTILE_USE_CBLAS
    // Check that matrix sizes fit proper types for underlying CUBLAS
    constexpr Scalar zero = 0, one = 1;
    enum starpu_data_access_mode C_mode;
    if(beta == zero)
    {
        C_mode = STARPU_W;
    }
    else if(beta == one)
    {
        if(redux != 0)
        {
            C_mode = STARPU_REDUX;
        }
        else
        {
            C_mode = static_cast<starpu_data_access_mode>(
                STARPU_RW | STARPU_COMMUTE);
        }
    }
    else
    {
        C_mode = STARPU_RW;
    }
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->transA = transA;
    args->transB = transB;
    args->m = m;
    args->n = n;
    args->k = k;
    args->batch = batch;
    args->alpha = alpha;
    args->beta = beta;
    // FLOPs calculation
    double nflops = 2 * m * n * k * batch;
    // Submit task
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &A }, { STARPU_R, &B }, { C_mode, &C } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class Gemm<std::tuple<nntile::fp64_t>>;
template class Gemm<std::tuple<nntile::fp32_t>>;
template class Gemm<std::tuple<nntile::fp32_fast_tf32_t>>;
template class Gemm<std::tuple<nntile::fp32_fast_fp16_t>>;
template class Gemm<std::tuple<nntile::fp32_fast_bf16_t>>;
template class Gemm<std::tuple<nntile::bf16_t>>;
template class Gemm<std::tuple<nntile::fp16_t>>;

//! Pack of gemm operations for different types
gemm_pack_t gemm;

} // namespace nntile::haul
