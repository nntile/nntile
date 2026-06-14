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

struct CublasGemmLayout
{
    cublasOperation_t trans_first = CUBLAS_OP_N;
    cublasOperation_t trans_second = CUBLAS_OP_N;
};

//! Map logical C-order GEMM operands to cuBLAS (row-major B_op @ A_op).
CublasGemmLayout c_order_gemm_layout(TransOp transA, TransOp transB)
{
    CublasGemmLayout layout;
    switch(transB.value)
    {
        case TransOp::NoTrans:
            switch(transA.value)
            {
                case TransOp::NoTrans:
                    layout.trans_first = CUBLAS_OP_T;
                    layout.trans_second = CUBLAS_OP_N;
                    break;
                case TransOp::Trans:
                default:
                    layout.trans_first = CUBLAS_OP_T;
                    layout.trans_second = CUBLAS_OP_T;
                    break;
            }
            break;
        case TransOp::Trans:
        default:
            switch(transA.value)
            {
                case TransOp::NoTrans:
                    layout.trans_first = CUBLAS_OP_N;
                    layout.trans_second = CUBLAS_OP_N;
                    break;
                case TransOp::Trans:
                default:
                    layout.trans_first = CUBLAS_OP_N;
                    layout.trans_second = CUBLAS_OP_T;
                    break;
            }
            break;
    }
    return layout;
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
    T *C
) noexcept
{
    // C-order layout matches kernel::cblas::gemm (row-major B_op @ A_op).
    const int M_out = n;
    const int N_out = m;
    const int K_inner = k;
    const int BATCH = batch;
    CublasGemmLayout layout = c_order_gemm_layout(transA, transB);
    int ld_first = 0;
    int ld_second = 0;
    switch(transB.value)
    {
        case TransOp::NoTrans:
            ld_first = M_out;
            ld_second = N_out;
            break;
        case TransOp::Trans:
        default:
            ld_first = K_inner;
            ld_second = N_out;
            break;
    }
    const int ldC = N_out;
    const long long int strideA =
        static_cast<long long int>(N_out) * K_inner;
    const long long int strideB =
        static_cast<long long int>(M_out) * K_inner;
    const long long int strideC =
        static_cast<long long int>(M_out) * N_out;
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
        layout.trans_second,
        layout.trans_first,
        N_out,
        M_out,
        K_inner,
        &alpha_,
        reinterpret_cast<const void *>(A),
        typeA,
        ld_second,
        strideA,
        reinterpret_cast<const void *>(B),
        typeB,
        ld_first,
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
