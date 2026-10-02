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

namespace
{

void c_order_gemm_ld(
    TransOp transA,
    TransOp transB,
    Index m,
    Index n,
    Index k,
    cublasOperation_t &transA_out,
    cublasOperation_t &transB_out,
    int &ldA,
    int &ldB)
{
    switch(transA.value)
    {
        case TransOp::NoTrans:
            transA_out = CUBLAS_OP_N;
            ldA = static_cast<int>(k);
            break;
        case TransOp::Trans:
        default:
            transA_out = CUBLAS_OP_T;
            ldA = static_cast<int>(m);
            break;
    }
    switch(transB.value)
    {
        case TransOp::NoTrans:
            transB_out = CUBLAS_OP_N;
            ldB = static_cast<int>(n);
            break;
        case TransOp::Trans:
        default:
            transB_out = CUBLAS_OP_T;
            ldB = static_cast<int>(k);
            break;
    }
}

} // namespace

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
    T *C,
    bool broadcast_a,
    bool broadcast_b
) noexcept
{
    // Row-major C(m,n) via cuBLAS: C^T = B^T * A^T in column-major layout.
    const int M = static_cast<int>(m);
    const int N = static_cast<int>(n);
    const int K = static_cast<int>(k);
    const int BATCH = static_cast<int>(batch);
    cublasOperation_t transA_ = CUBLAS_OP_N;
    cublasOperation_t transB_ = CUBLAS_OP_N;
    int ldA = 0;
    int ldB = 0;
    c_order_gemm_ld(transA, transB, m, n, k, transA_, transB_, ldA, ldB);
    const int ldC = N;
    const long long int strideA = broadcast_a ? 0LL :
        static_cast<long long int>(m) * k;
    const long long int strideB = broadcast_b ? 0LL :
        static_cast<long long int>(k) * n;
    const long long int strideC =
        static_cast<long long int>(m) * n;
    scalar_t<T> alpha_=alpha, beta_=beta;

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
        computeType = CUBLAS_COMPUTE_32F;
    }

    // Call corresponding CUBLAS routine
    cublasGemmStridedBatchedEx(
        handle,
        transB_,
        transA_,
        N,
        M,
        K,
        &alpha_,
        reinterpret_cast<const void *>(B),
        typeB,
        ldB,
        strideB,
        reinterpret_cast<const void *>(A),
        typeA,
        ldA,
        strideA,
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
    fp64_t *C,
    bool broadcast_a,
    bool broadcast_b) noexcept;

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
    fp32_t *C,
    bool broadcast_a,
    bool broadcast_b) noexcept;

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
    fp32_fast_tf32_t *C,
    bool broadcast_a,
    bool broadcast_b) noexcept;

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
    fp32_fast_fp16_t *C,
    bool broadcast_a,
    bool broadcast_b) noexcept;

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
    fp32_fast_bf16_t *C,
    bool broadcast_a,
    bool broadcast_b) noexcept;

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
    bf16_t *C,
    bool broadcast_a,
    bool broadcast_b) noexcept;

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
    fp16_t *C,
    bool broadcast_a,
    bool broadcast_b) noexcept;

} // namespace nntile:kernel::cublas
#endif // NNTILE_USE_CUDA
