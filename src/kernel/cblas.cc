/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/cblas.cc
 * Wrappers for CBLAS low-level routines
 *
 * @version 1.1.0
 * */

// Corresponding header
#include <nntile/kernel/cblas.hh>

// Only include the rest of the file if CBLAS is enabled
#ifdef NNTILE_USE_CBLAS

//! @namespace nntile::kernel::cblas
/*! Wrappers for CBLAS low-level routines
 * */
namespace nntile::kernel::cblas
{

// GEMM operation implementation
template<typename T>
void gemm(
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
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Convert values to CBLAS types
    CBLAS_INT M=m, N=n, K=k, ldA, ldB, ldC=M;
    // Convert transposition operation flags
    CBLAS_TRANSPOSE transA_, transB_;
    switch(transA.value)
    {
        case TransOp::NoTrans:
            transA_ = CblasNoTrans;
            ldA = M;
            break;
        case TransOp::Trans:
        default:
            transA_ = CblasTrans;
            ldA = K;
            break;
    }
    switch(transB.value)
    {
        case TransOp::NoTrans:
            transB_ = CblasNoTrans;
            ldB = K;
            break;
        case TransOp::Trans:
        default:
            transB_ = CblasTrans;
            ldB = N;
            break;
    }
    // Call corresponding CBLAS routine for every batch
    Index A_offset = m * k, B_offset = n * k, C_offset = m * n;
    for(Index i = 0; i < batch; ++i)
    {
        if constexpr(std::is_same_v<T, fp64_t>) // Double precision
        {
            cblas_dgemm(
                CblasColMajor,
                transA_,
                transB_,
                M,
                N,
                K,
                alpha, // alpha is upcasted to double
                reinterpret_cast<const double *>(A),
                ldA,
                reinterpret_cast<const double *>(B),
                ldB,
                beta, // beta is upcasted to double
                reinterpret_cast<double *>(C),
                ldC
            );
        }
        else if constexpr(std::is_same_v<T, fp32_t>) // Single precision
        {
            cblas_sgemm(
                CblasColMajor,
                transA_,
                transB_,
                M,
                N,
                K,
                alpha,
                reinterpret_cast<const float *>(A),
                ldA,
                reinterpret_cast<const float *>(B),
                ldB,
                beta,
                reinterpret_cast<float *>(C),
                ldC
            );
        }
        // Other precisions not supported, better to report it, but we do not
        // Loop to the next batch
        A += A_offset;
        B += B_offset;
        C += C_offset;
    }
#endif // STARPU_SIMGRID
}

// Explicit instantiation
template void gemm<fp64_t>(
    TransOp transA, TransOp transB, Index m, Index n, Index k, Index batch,
    Scalar alpha, const fp64_t *A, const fp64_t *B, Scalar beta, fp64_t *C) noexcept;

template void gemm<fp32_t>(
    TransOp transA, TransOp transB, Index m, Index n, Index k, Index batch,
    Scalar alpha, const fp32_t *A, const fp32_t *B, Scalar beta, fp32_t *C) noexcept;

} // namespace nntile:kernel::cblas
#endif // NNTILE_USE_CBLAS
