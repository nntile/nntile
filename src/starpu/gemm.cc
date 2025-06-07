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
#include "nntile/starpu/gemm.hh"

// Standard libraries
#include <cstdlib>

// Third-party headers
#include <starpu_cublas_v2.h>

// Other NNTile headers
#include "nntile/kernel/cblas.hh"
#include "nntile/kernel/cublas.hh"

namespace nntile::starpu
{

//! Constructor
template<typename T>
Gemm<std::tuple<T>>::Gemm():
    codelet("nntile_gemm", footprint, cpu_funcs, cuda_funcs)
{
    // Modes are not fixed, they are decided during runtime by default
    // Tell StarPU that we do not support all types
#ifdef NNTILE_USE_CBLAS
    if constexpr (!kernel::cblas::gemm_is_supported<T>)
    {
        codelet.cpu_funcs[0] = nullptr;
        codelet.where_default = codelet.where_default ^ STARPU_CPU;
        codelet.where = codelet.where_default;
    }
#endif // NNTILE_USE_CBLAS
}

#ifdef NNTILE_USE_CBLAS // CPU implementation requires CBLAS
//! GEMM for contiguous matrices without padding through StarPU buffers
template<typename T>
void Gemm<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    // Launch kernel
    const T *A = interfaces[0]->get_ptr<T>();
    const T *B = interfaces[1]->get_ptr<T>();
    T *C = interfaces[2]->get_ptr<T>();
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
#endif // STARPU_SIMGRID
}
#endif // NNTILE_USE_CBLAS

#ifdef NNTILE_USE_CUDA // CUDA implementation requires cuBLAS
//! GEMM for contiguous matrices without padding through StarPU buffers
template<typename T>
void Gemm<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    // Launch kernel
    const T *A = interfaces[0]->get_ptr<T>();
    const T *B = interfaces[1]->get_ptr<T>();
    T *C = interfaces[2]->get_ptr<T>();
    // Get cuBLAS handle and CUDA stream
    cublasHandle_t handle = starpu_cublas_get_local_handle();
    cudaStream_t stream = starpu_cuda_get_local_stream();
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
#endif // STARPU_SIMGRID
}
#endif //NNTILE_USE_CUDA

//! Footprint for GEMM tasks that depends on transA, transB, M, N, K, batch and alpha
template<typename T>
uint32_t Gemm<std::tuple<T>>::footprint(struct starpu_task *task)
{
    // Get arguments
    auto args = reinterpret_cast<args_t *>(task->cl_arg);
    // In case alpha is zero, entire gemm is unnecessary so it is better to
    // give it a different footprint since gemm time will be totally different
    uint32_t hash = args->alpha == Scalar{0} ? -1 : 0;
    // Single codelet is used for all combinations of transA and transB
    hash = starpu_hash_crc32c_be_n(&args->transA, sizeof(args->transA), hash);
    hash = starpu_hash_crc32c_be_n(&args->transB, sizeof(args->transB), hash);
    // Apply hash over parameters M, N and K. This way if we swap values of M,
    // N and K total size of buffers will remain the same, but the footprint
    // will be different
    hash = starpu_hash_crc32c_be_n(&args->m, sizeof(args->m), hash);
    hash = starpu_hash_crc32c_be_n(&args->n, sizeof(args->n), hash);
    hash = starpu_hash_crc32c_be_n(&args->k, sizeof(args->k), hash);
    hash = starpu_hash_crc32c_be_n(&args->batch, sizeof(args->batch), hash);
    return hash;
}

template<typename T>
void Gemm<std::tuple<T>>::submit(const TransOp &transA, const TransOp &transB, Index m, Index n,
        Index k, Index batch, Scalar alpha, Handle A, Handle B, Scalar beta,
        Handle C, int redux)
{
    // Check that matrix sizes fit proper types for underlying CBLAS
#ifdef NNTILE_USE_CBLAS
#ifndef STARPU_SIMGRID
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
#endif // STARPU_SIMGRID
#endif // NNTILE_USE_CBLAS
    // Check that matrix sizes fit proper types for underlying CUBLAS
#ifdef NNTILE_USE_CUDA
#ifndef STARPU_SIMGRID
    if(static_cast<int>(m) != m)
    {
        throw std::runtime_error("GEMM size M does not fit int");
    }
    if(static_cast<int>(n) != n)
    {
        throw std::runtime_error("GEMM size N does not fit int");
    }
    if(static_cast<int>(k) != k)
    {
        throw std::runtime_error("GEMM size K does not fit int");
    }
#endif // STARPU_SIMGRID
#endif // NNTILE_USE_CUDA
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
    int ret = starpu_task_insert(&codelet,
            STARPU_R, A.get(),
            STARPU_R, B.get(),
            C_mode, C.get(),
            STARPU_CL_ARGS, args, sizeof(*args),
            STARPU_FLOPS, nflops,
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in gemm task submission");
    }
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

//! Pack of gemm operations for different types
gemm_pack_t gemm;

} // namespace nntile::starpu
