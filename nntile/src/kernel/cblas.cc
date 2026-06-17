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

namespace
{

void c_order_gemm_ld(
    TransOp transA,
    TransOp transB,
    Index m,
    Index n,
    Index k,
    CBLAS_TRANSPOSE &transA_out,
    CBLAS_TRANSPOSE &transB_out,
    CBLAS_INT &ldA,
    CBLAS_INT &ldB)
{
    switch(transA.value)
    {
        case TransOp::NoTrans:
            transA_out = CblasNoTrans;
            ldA = static_cast<CBLAS_INT>(k);
            break;
        case TransOp::Trans:
        default:
            transA_out = CblasTrans;
            ldA = static_cast<CBLAS_INT>(m);
            break;
    }
    switch(transB.value)
    {
        case TransOp::NoTrans:
            transB_out = CblasNoTrans;
            ldB = static_cast<CBLAS_INT>(n);
            break;
        case TransOp::Trans:
        default:
            transB_out = CblasTrans;
            ldB = static_cast<CBLAS_INT>(k);
            break;
    }
}

} // namespace

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
    T *C,
    bool broadcast_a,
    bool broadcast_b
) noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    const CBLAS_INT M = static_cast<CBLAS_INT>(m);
    const CBLAS_INT N = static_cast<CBLAS_INT>(n);
    const CBLAS_INT K = static_cast<CBLAS_INT>(k);
    CBLAS_TRANSPOSE transA_ = CblasNoTrans;
    CBLAS_TRANSPOSE transB_ = CblasNoTrans;
    CBLAS_INT ldA = 0;
    CBLAS_INT ldB = 0;
    c_order_gemm_ld(transA, transB, m, n, k, transA_, transB_, ldA, ldB);
    const CBLAS_INT ldC = N;
    const Index a_stride = static_cast<Index>(m) * k;
    const Index b_stride = static_cast<Index>(k) * n;
    const Index c_stride = static_cast<Index>(m) * n;
    for(Index i = 0; i < batch; ++i)
    {
        if constexpr(std::is_same_v<T, fp64_t>) // Double precision
        {
            cblas_dgemm(
                CblasRowMajor,
                transA_,
                transB_,
                M,
                N,
                K,
                alpha,
                reinterpret_cast<const double *>(A),
                ldA,
                reinterpret_cast<const double *>(B),
                ldB,
                beta,
                reinterpret_cast<double *>(C),
                ldC
            );
        }
        else if constexpr(std::is_same_v<T, fp32_t>) // Single precision
        {
            cblas_sgemm(
                CblasRowMajor,
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
        if(!broadcast_b)
        {
            B += b_stride;
        }
        if(!broadcast_a)
        {
            A += a_stride;
        }
        C += c_stride;
    }
#endif // STARPU_SIMGRID
}

// Explicit instantiation
template void gemm<fp64_t>(
    TransOp transA, TransOp transB, Index m, Index n, Index k, Index batch,
    Scalar alpha, const fp64_t *A, const fp64_t *B, Scalar beta, fp64_t *C,
    bool broadcast_a, bool broadcast_b) noexcept;

template void gemm<fp32_t>(
    TransOp transA, TransOp transB, Index m, Index n, Index k, Index batch,
    Scalar alpha, const fp32_t *A, const fp32_t *B, Scalar beta, fp32_t *C,
    bool broadcast_a, bool broadcast_b) noexcept;

} // namespace nntile:kernel::cblas
#endif // NNTILE_USE_CBLAS
