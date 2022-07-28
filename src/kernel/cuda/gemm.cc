/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/cuda/gemm.cc
 * GEMM operation for Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#include "nntile/kernel/cuda/gemm.hh"
#include <starpu_data_interfaces.h>
#include <cublas_v2.h>
#include <starpu_cublas_v2.h>

namespace nntile
{

// Overloaded call to cuBLAS GEMM
static inline
void cublas_gemm(cublasHandle_t handle, cublasOperation_t transA,
        cublasOperation_t transB, int M, int N, int K, fp32_t alpha,
        const fp32_t *A, int ldA, const fp32_t *B, int ldB, fp32_t beta,
        fp32_t *C, int ldC)
{
    cublasSgemm(handle, transA, transB, M, N, K, &alpha, A, ldA, B, ldB, &beta,
            C, ldC);
}

// Overloaded call to cuBLAS GEMM
static inline
void cublas_gemm(cublasHandle_t handle, cublasOperation_t transA,
        cublasOperation_t transB, int M, int N, int K, fp64_t alpha,
        const fp64_t *A, int ldA, const fp64_t *B, int ldB, fp64_t beta,
        fp64_t *C, int ldC)
{
    cublasDgemm(handle, transA, transB, M, N, K, &alpha, A, ldA, B, ldB, &beta,
            C, ldC);
}

//! GEMM for contiguous matrices without padding
template<typename T>
void gemm_kernel_cublas(TransOp transA, TransOp transB, Index m, Index n,
        Index k, T alpha, const T *A, const T *B, T beta, T *C)
{
    // It is OK to convert values as it was checked during task submission
    int M=m, N=n, K=k, ldA, ldB, ldC=M;
    cublasOperation_t transA_, transB_;
    // Convert other values to CBLAS types
    switch(transA_value)
    {
        case TransOp::NoTrans:
            transA_ = CUBLAS_OP_N;
            ldA = M;
            break;
        // This parameter was already checked in gemm_check_opA_opB
        //case TransOp::Trans:
        default:
            transA_ = CUBLAS_OP_T;
            ldA = K;
    }
    switch(transB_value)
    {
        case TransOp::NoTrans:
            transB_ = CUBLAS_OP_N;
            ldB = K;
            break;
        // This parameter was already checked in gemm_check_opA_opB
        //case TransOp::Trans:
        default:
            transB_ = CUBLAS_OP_T;
            ldB = N;
    }
    cublasHandle_t handle = starpu_cublas_get_local_handle();
    cudaStream_t stream = starpu_cuda_get_local_stream();
    cublasSetStream(handle, stream);
    // Call corresponding cuBLAS routine
    cublas_gemm(handle, transA_, transB_, M, N, K, alpha, A, ldA, B, ldB, beta,
            C, M);
}

//! GEMM for contiguous matrices without padding through StarPU buffers
template<typename T>
void gemm_starpu_cuda(void *buffers[], void *cl_args)
{
    TransOp transA(TransOp::Trans), transB(transA);
    Index m, n, k;
    T alpha, beta;
    starpu_codelet_unpack_args(cl_args, &transA.value, &transB.value, &m, &n,
            &k, &alpha, &beta);
    const T *A = reinterpret_cast<T *>(STARPU_NDIM_GET_PTR(buffers[0]));
    const T *B = reinterpret_cast<T *>(STARPU_NDIM_GET_PTR(buffers[1]));
    T *C = reinterpret_cast<T *>(STARPU_NDIM_GET_PTR(buffers[2]));
    gemm_kernel_cublas<T>(transA, transB, m, n, k, alpha, A, B, beta, C);
}

// Explicit instantiation of templates
template
void gemm_starpu_cuda<fp32_t>(void *buffers[], void *cl_args);

template
void gemm_starpu_cuda<fp64_t>(void *buffers[], void *cl_args);

} // namespace nntile

