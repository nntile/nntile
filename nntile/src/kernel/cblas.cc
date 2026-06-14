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

struct CblasGemmLayout
{
    CBLAS_TRANSPOSE trans_first = CblasNoTrans;
    CBLAS_TRANSPOSE trans_second = CblasNoTrans;
    CBLAS_INT ld_first = 0;
    CBLAS_INT ld_second = 0;
};

//! Map logical C-order GEMM operands to row-major CBLAS call B_op @ A_op.
CblasGemmLayout c_order_gemm_layout(TransOp transA, TransOp transB)
{
    CblasGemmLayout layout;
    switch(transB.value)
    {
        case TransOp::NoTrans:
            layout.ld_first = 0; // set from n below
            switch(transA.value)
            {
                case TransOp::NoTrans:
                    layout.trans_first = CblasTrans;
                    layout.trans_second = CblasNoTrans;
                    break;
                case TransOp::Trans:
                default:
                    layout.trans_first = CblasTrans;
                    layout.trans_second = CblasTrans;
                    break;
            }
            break;
        case TransOp::Trans:
        default:
            switch(transA.value)
            {
                case TransOp::NoTrans:
                    layout.trans_first = CblasNoTrans;
                    layout.trans_second = CblasNoTrans;
                    break;
                case TransOp::Trans:
                default:
                    layout.trans_first = CblasNoTrans;
                    layout.trans_second = CblasTrans;
                    break;
            }
            break;
    }
    return layout;
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
    T *C
) noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    const CBLAS_INT M_out = static_cast<CBLAS_INT>(n);
    const CBLAS_INT N_out = static_cast<CBLAS_INT>(m);
    const CBLAS_INT K_inner = static_cast<CBLAS_INT>(k);
    CblasGemmLayout layout = c_order_gemm_layout(transA, transB);
    CBLAS_INT ld_first = 0;
    CBLAS_INT ld_second = 0;
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
    const CBLAS_INT ldC = N_out;
    const Index first_offset = static_cast<Index>(M_out) * K_inner;
    const Index second_offset = static_cast<Index>(N_out) * K_inner;
    const Index c_offset = static_cast<Index>(M_out) * N_out;
    for(Index i = 0; i < batch; ++i)
    {
        if constexpr(std::is_same_v<T, fp64_t>) // Double precision
        {
            cblas_dgemm(
                CblasRowMajor,
                layout.trans_first,
                layout.trans_second,
                M_out,
                N_out,
                K_inner,
                alpha,
                reinterpret_cast<const double *>(B),
                ld_first,
                reinterpret_cast<const double *>(A),
                ld_second,
                beta,
                reinterpret_cast<double *>(C),
                ldC
            );
        }
        else if constexpr(std::is_same_v<T, fp32_t>) // Single precision
        {
            cblas_sgemm(
                CblasRowMajor,
                layout.trans_first,
                layout.trans_second,
                M_out,
                N_out,
                K_inner,
                alpha,
                reinterpret_cast<const float *>(B),
                ld_first,
                reinterpret_cast<const float *>(A),
                ld_second,
                beta,
                reinterpret_cast<float *>(C),
                ldC
            );
        }
        B += first_offset;
        A += second_offset;
        C += c_offset;
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
