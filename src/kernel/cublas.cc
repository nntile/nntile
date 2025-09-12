/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/cublas.cc
 * Wrappers for CUBLAS low-level routines
 *
 * @version 1.1.0
 * */

// Corresponding header
#include <nntile/kernel/cublas.hh>

// Only include the rest of the file if CUDA (and CUBLAS) is enabled
#ifdef NNTILE_USE_CUDA

//! @namespace nntile::kernel::cublas
/*! Wrappers for CUBLAS low-level routines
 * */
namespace nntile::kernel::cublas
{

//! Helper type to get type of scalars for cublasGemmEx
/*! Currently, it coincides with our representation type, but it will be wrong
 *  once we add fp16 support
 * */
template<typename T>
using scalar_t = typename T::repr_t;

// GEMM operation implementation
template<typename T>
void gemm(
    cublasHandle_t handle,
    TransOp transA,
    TransOp transB,
    Index m,
    Index n,
    Index k,
    Index batch,
    Scalar alpha,
    const T *A,
    const T *B,
    Scalar beta,
    T *C
) noexcept
{
    // Convert values to CUBLAS types
    int M=m, N=n, K=k, ldA, ldB, ldC=M;
    int BATCH=batch;
    long long int strideA=m*k, strideB=n*k, strideC=m*n;
    scalar_t<T> alpha_=alpha, beta_=beta;

    // Convert transposition operation flags
    cublasOperation_t transA_, transB_;
    switch(transA.value)
    {
        case TransOp::NoTrans:
            transA_ = CUBLAS_OP_N;
            ldA = M;
            break;
        case TransOp::Trans:
        default:
            transA_ = CUBLAS_OP_T;
            ldA = K;
            break;
    }
    switch(transB.value)
    {
        case TransOp::NoTrans:
            transB_ = CUBLAS_OP_N;
            ldB = K;
            break;
        case TransOp::Trans:
        default:
            transB_ = CUBLAS_OP_T;
            ldB = N;
            break;
    }

    // Find out cublasGemmEx specific parameters
    cudaDataType_t typeA, typeB, typeC;
    cublasComputeType_t computeType;
    constexpr cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    if constexpr(std::is_same_v<T, fp64_t>)
    {
        typeA = CUDA_R_64F;
        typeB = CUDA_R_64F;
        typeC = CUDA_R_64F;
        computeType = CUBLAS_COMPUTE_64F;
    }
    else if constexpr(std::is_same_v<T, fp32_t>)
    {
        typeA = CUDA_R_32F;
        typeB = CUDA_R_32F;
        typeC = CUDA_R_32F;
        computeType = CUBLAS_COMPUTE_32F;
    }
    else if constexpr(std::is_same_v<T, fp32_fast_tf32_t>)
    {
        typeA = CUDA_R_32F;
        typeB = CUDA_R_32F;
        typeC = CUDA_R_32F;
        computeType = CUBLAS_COMPUTE_32F_FAST_TF32;
    }
    else if constexpr(std::is_same_v<T, fp32_fast_fp16_t>)
    {
        typeA = CUDA_R_32F;
        typeB = CUDA_R_32F;
        typeC = CUDA_R_32F;
        computeType = CUBLAS_COMPUTE_32F_FAST_16F;
    }
    else if constexpr(std::is_same_v<T, fp32_fast_bf16_t>)
    {
        typeA = CUDA_R_32F;
        typeB = CUDA_R_32F;
        typeC = CUDA_R_32F;
        computeType = CUBLAS_COMPUTE_32F_FAST_16BF;
    }
    else if constexpr(std::is_same_v<T, bf16_t>)
    {
        typeA = CUDA_R_16BF;
        typeB = CUDA_R_16BF;
        typeC = CUDA_R_16BF;
        computeType = CUBLAS_COMPUTE_32F;
    }
    else if constexpr(std::is_same_v<T, fp16_t>)
    {
        typeA = CUDA_R_16F;
        typeB = CUDA_R_16F;
        typeC = CUDA_R_16F;
        computeType = CUBLAS_COMPUTE_16F;
    }

    // Call corresponding CUBLAS routine
    cublasGemmStridedBatchedEx(
        handle,
        transA_,
        transB_,
        M,
        N,
        K,
        &alpha_,
        reinterpret_cast<const void *>(A),
        typeA,
        ldA,
        strideA,
        reinterpret_cast<const void *>(B),
        typeB,
        ldB,
        strideB,
        &beta_,
        reinterpret_cast<void *>(C),
        typeC,
        ldC,
        strideC,
        BATCH,
        computeType,
        algo
    );
}

// Explicit instantiation
template void gemm<fp64_t>(
    cublasHandle_t handle,
    TransOp transA,
    TransOp transB,
    Index m,
    Index n,
    Index k,
    Index batch,
    Scalar alpha,
    const fp64_t *A,
    const fp64_t *B,
    Scalar beta,
    fp64_t *C
) noexcept;

template void gemm<fp32_t>(
    cublasHandle_t handle,
    TransOp transA,
    TransOp transB,
    Index m,
    Index n,
    Index k,
    Index batch,
    Scalar alpha,
    const fp32_t *A,
    const fp32_t *B,
    Scalar beta,
    fp32_t *C
) noexcept;

template void gemm<fp32_fast_tf32_t>(
    cublasHandle_t handle,
    TransOp transA,
    TransOp transB,
    Index m,
    Index n,
    Index k,
    Index batch,
    Scalar alpha,
    const fp32_fast_tf32_t *A,
    const fp32_fast_tf32_t *B,
    Scalar beta,
    fp32_fast_tf32_t *C
) noexcept;

template void gemm<fp32_fast_fp16_t>(
    cublasHandle_t handle,
    TransOp transA,
    TransOp transB,
    Index m,
    Index n,
    Index k,
    Index batch,
    Scalar alpha,
    const fp32_fast_fp16_t *A,
    const fp32_fast_fp16_t *B,
    Scalar beta,
    fp32_fast_fp16_t *C
) noexcept;

template void gemm<fp32_fast_bf16_t>(
    cublasHandle_t handle,
    TransOp transA,
    TransOp transB,
    Index m,
    Index n,
    Index k,
    Index batch,
    Scalar alpha,
    const fp32_fast_bf16_t *A,
    const fp32_fast_bf16_t *B,
    Scalar beta,
    fp32_fast_bf16_t *C
) noexcept;

template void gemm<bf16_t>(
    cublasHandle_t handle,
    TransOp transA,
    TransOp transB,
    Index m,
    Index n,
    Index k,
    Index batch,
    Scalar alpha,
    const bf16_t *A,
    const bf16_t *B,
    Scalar beta,
    bf16_t *C
) noexcept;

template void gemm<fp16_t>(
    cublasHandle_t handle,
    TransOp transA,
    TransOp transB,
    Index m,
    Index n,
    Index k,
    Index batch,
    Scalar alpha,
    const fp16_t *A,
    const fp16_t *B,
    Scalar beta,
    fp16_t *C
) noexcept;

} // namespace nntile:kernel::cblas
#endif // NNTILE_USE_CBLAS
