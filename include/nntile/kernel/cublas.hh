/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/cublas.hh
 * Wrappers for CUBLAS low-level routines
 *
 * @version 1.1.0
 * */

#pragma once

// Compile-time definitions
#include <nntile/defs.h>

// Only include the rest of the file if CUDA (and CUBLAS) is enabled
#ifdef NNTILE_USE_CUDA

// Third-party headers
#include <cublas_v2.h>

// Other NNTile headers
#include <nntile/base_types.hh>
#include <nntile/constants.hh>

//! @namespace nntile::kernel::cublas
/*! Wrappers for CUBLAS low-level routines
 * */
namespace nntile::kernel::cublas
{

//! GEMM operation
/*! Performs the matrix-matrix:
 * C = alpha * op(A) * op(B) + beta * C,
 * where op(A) and op(B) are matrices A and B, transposed or not,
 * respectively, and C is the result matrix. All matrices are stored in a
 * packed format without padding. Therefore, leading dimension of each matrix
 * is deduced from parameters M, N, K and transposition operation flags.
 *
 * @param[in] handle: CUBLAS handle
 * @param[in] transA: Transposition operation for matrix A
 * @param[in] transB: Transposition operation for matrix B
 * @param[in] M: Number of rows in matrix op(A)
 * @param[in] N: Number of columns in matrix op(B)
 * @param[in] K: Number of columns in matrix A and number of rows in matrix B
 * @param[in] alpha: Scalar multiplier for matrix A
 * @param[in] A: Pointer to matrix A
 * @param[in] B: Pointer to matrix B
 * @param[in] beta: Scalar multiplier for matrix C
 * @param[in] C: Pointer to matrix C
 * */
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
) noexcept;

} // namespace nntile:kernel::cublas
#endif // NNTILE_USE_CUDA
