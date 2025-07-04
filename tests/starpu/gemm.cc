/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/starpu/gemm.cc
 * GEMM operation for StarPU buffers
 *
 * @version 1.1.0
 * */

#include "nntile/context.hh"
#include "nntile/starpu/gemm.hh"
#include "nntile/kernel/cblas.hh"
#include "nntile/kernel/cublas.hh"
#include "../testing.hh"
#include <vector>
#include <stdexcept>
#include <iostream>

using namespace nntile;
using namespace nntile::starpu;

#ifdef NNTILE_USE_CBLAS
// Overloaded call to CBLAS GEMM
static inline
void cblas_gemm(CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
        CBLAS_INT M, CBLAS_INT N, CBLAS_INT K, float alpha, const fp32_t *A,
        CBLAS_INT ldA, const fp32_t *B, CBLAS_INT ldB, float beta, fp32_t *C,
        CBLAS_INT ldC)
    noexcept
{
    cblas_sgemm(CblasColMajor, transA, transB, M, N, K, alpha,
            (const float *)A, ldA, (const float *)B, ldB, beta, (float *)C, ldC);
}

// Overloaded call to CBLAS GEMM
static inline
void cblas_gemm(CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
        CBLAS_INT M, CBLAS_INT N, CBLAS_INT K, double alpha, const fp64_t *A,
        CBLAS_INT ldA, const fp64_t *B, CBLAS_INT ldB, double beta, fp64_t *C,
        CBLAS_INT ldC)
    noexcept
{
    cblas_dgemm(CblasColMajor, transA, transB, M, N, K, alpha,
            (const double *)A, ldA, (const double *)B, ldB, beta, (double *)C, ldC);
}

template<typename T>
void validate_cpu(TransOp transA, TransOp transB, Index m, Index n, Index k,
        Index batch, Scalar alpha, Scalar beta)
{
    using Y = typename T::repr_t;
    // Init all the data
    std::vector<T> A(m*k*batch), B(n*k*batch), C(m*n*batch);
    for(Index i = 0; i < A.size(); ++i)
    {
        A[i] = Y(i+1);
    }
    for(Index i = 0; i < B.size(); ++i)
    {
        B[i] = Y(-i-1);
    }
    for(Index i = 0; i < C.size(); ++i)
    {
        C[i] = Y(2*i+1);
    }
    // Create copy of C
    std::vector<T> C2(C);
    // Launch low-level kernel
    CBLAS_TRANSPOSE transA_, transB_;
    Index ldA, ldB;
    switch(transA.value)
    {
        case TransOp::NoTrans:
            transA_ = CblasNoTrans;
            ldA = m;
            break;
        case TransOp::Trans:
            transA_ = CblasTrans;
            ldA = k;
    }
    switch(transB.value)
    {
        case TransOp::NoTrans:
            transB_ = CblasNoTrans;
            ldB = k;
            break;
        case TransOp::Trans:
            transB_ = CblasTrans;
            ldB = n;
    }
    std::cout << "Run cblas_gemm<" << T::short_name << ">\n";
    for(Index b = 0; b < batch; ++b)
    {
        cblas_gemm(transA_, transB_, m, n, k, alpha, &A[b*m*k], ldA, &B[b*n*k],
                ldB, beta, &C[b*m*n], m);
    }
    // Check by actually submitting a task
    VariableHandle A_handle(&A[0], sizeof(T)*A.size()),
        B_handle(&B[0], sizeof(T)*B.size()),
        C2_handle(&C2[0], sizeof(T)*C.size());
    gemm.restrict_where(STARPU_CPU);
    std::cout << "Run starpu::gemm::submit<" << T::short_name << "> restricted to CPU\n";
    gemm.submit<std::tuple<T>>(transA, transB, m, n, k, batch, alpha, A_handle, B_handle,
            beta, C2_handle);
    starpu_task_wait_for_all();
    C2_handle.unregister();
    // Check result
    for(Index i = 0; i < C.size(); ++i)
    {
        TEST_ASSERT((Y(C[i])-Y(C2[i])) <= T::epsilon * std::abs(Y(C[i])));
    }
    std::cout << "OK: starpu::gemm::submit<" << T::short_name << "> restricted to CPU\n";
}

template<typename T>
void validate_cpu_many()
{
    TransOp opT(TransOp::Trans), opN(TransOp::NoTrans);
    TransOp trans[2] = {opN, opT};
    Scalar alpha[3] = {0, 1, -3};
    Scalar beta[3] = {0, 1, 2};
    Index batch[3] = {1, 2, 100};
    for(auto transA: trans)
    {
        for(auto transB: trans)
        {
            for(Scalar a: alpha)
            {
                for(Scalar b: beta)
                {
                    for(auto nb: batch)
                    {
                        validate_cpu<T>(transA, transB, 10, 6, 3, nb, a, b);
                    }
                }
            }
        }
    }
}
#endif // NNTILE_USE_CBLAS

#ifdef NNTILE_USE_CUDA
// Overloaded call to cuBLAS GEMM
static inline
void cublas_gemm(cublasHandle_t handle, cublasOperation_t transA,
        cublasOperation_t transB, int M, int N, int K, float alpha,
        const fp32_t *A, int ldA, const fp32_t *B, int ldB, float beta,
        fp32_t *C, int ldC)
    noexcept
{
    cublasSgemm(handle, transA, transB, M, N, K, &alpha, (const float *)A, ldA,
            (const float *)B, ldB, &beta, (float *)C, ldC);
}

// Overloaded call to cuBLAS GEMM
static inline
void cublas_gemm(cublasHandle_t handle, cublasOperation_t transA,
        cublasOperation_t transB, int M, int N, int K, double alpha,
        const fp64_t *A, int ldA, const fp64_t *B, int ldB, double beta,
        fp64_t *C, int ldC)
    noexcept
{
    cublasDgemm(handle, transA, transB, M, N, K, &alpha, (const double *)A, ldA,
            (const double *)B, ldB, &beta, (double *)C, ldC);
}

template<typename T>
void validate_cuda(TransOp transA, TransOp transB, Index m, Index n, Index k,
        Index batch, Scalar alpha, Scalar beta)
{
    using Y = typename T::repr_t;
    // Get a StarPU CUDA worker (to perform computations on the same device)
    int cuda_worker_id = starpu_worker_get_by_type(STARPU_CUDA_WORKER, 0);
    // Choose worker CUDA device
    int dev_id = starpu_worker_get_devid(cuda_worker_id);
    cudaError_t cuda_err = cudaSetDevice(dev_id);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // cuBLAS handle
    cublasHandle_t cublas;
    cublasStatus_t cublas_err = cublasCreate(&cublas);
    TEST_ASSERT(cublas_err == CUBLAS_STATUS_SUCCESS);
    // Create CUDA stream and make cuBLAS use it
    cudaStream_t stream;
    cuda_err = cudaStreamCreate(&stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cublas_err = cublasSetStream(cublas, stream);
    TEST_ASSERT(cublas_err == CUBLAS_STATUS_SUCCESS);
    // Init all the data
    std::vector<T> A(m*k*batch), B(n*k*batch), C(m*n*batch);
    for(Index i = 0; i < A.size(); ++i)
    {
        A[i] = Y(i+1);
    }
    for(Index i = 0; i < B.size(); ++i)
    {
        B[i] = Y(-i-1);
    }
    for(Index i = 0; i < C.size(); ++i)
    {
        C[i] = Y(2*i+1);
    }
    // Create copy of C
    std::vector<T> C2(C);
    // Launch low-level kernel
    cublasOperation_t transA_, transB_;
    Index ldA, ldB;
    switch(transA.value)
    {
        case TransOp::NoTrans:
            transA_ = CUBLAS_OP_N;
            ldA = m;
            break;
        case TransOp::Trans:
            transA_ = CUBLAS_OP_T;
            ldA = k;
    }
    switch(transB.value)
    {
        case TransOp::NoTrans:
            transB_ = CUBLAS_OP_N;
            ldB = k;
            break;
        case TransOp::Trans:
            transB_ = CUBLAS_OP_T;
            ldB = n;
    }
    T *dev_A, *dev_B, *dev_C;
    cuda_err = cudaMalloc(&dev_A, sizeof(T)*A.size());
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMalloc(&dev_B, sizeof(T)*B.size());
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMalloc(&dev_C, sizeof(T)*C.size());
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_A, &A[0], sizeof(T)*A.size(),
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_B, &B[0], sizeof(T)*B.size(),
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_C, &C[0], sizeof(T)*C.size(),
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    std::cout << "Run cublas_gemm<" << T::short_name << ">\n";
    for(Index b = 0; b < batch; ++b)
    {
        cublas_gemm(cublas, transA_, transB_, m, n, k, alpha, &dev_A[b*m*k],
                ldA, &dev_B[b*n*k], ldB, beta, &dev_C[b*m*n], m);
    }
    // Wait for result and destroy cublas handle and stream
    cuda_err = cudaStreamSynchronize(stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cublas_err = cublasDestroy(cublas);
    TEST_ASSERT(cublas_err == CUBLAS_STATUS_SUCCESS)
    cuda_err = cudaStreamDestroy(stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Copy result back to CPU
    cuda_err = cudaMemcpy(&C[0], dev_C, sizeof(T)*C.size(),
            cudaMemcpyDeviceToHost);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Deallocate CUDA memory
    cuda_err = cudaFree(dev_A);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaFree(dev_B);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaFree(dev_C);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Check by actually submitting a task
    VariableHandle A_handle(&A[0], sizeof(T)*A.size()),
        B_handle(&B[0], sizeof(T)*B.size()),
        C2_handle(&C2[0], sizeof(T)*C.size());
    gemm.restrict_where(STARPU_CUDA);
    std::cout << "Run starpu::gemm::submit<" << T::short_name << "> restricted to CUDA\n";
    gemm.submit<std::tuple<T>>(transA, transB, m, n, k, batch, alpha, A_handle, B_handle,
            beta, C2_handle);
    starpu_task_wait_for_all();
    C2_handle.unregister();
    // Check result
    for(Index i = 0; i < C.size(); ++i)
    {
        TEST_ASSERT((Y(C[i])-Y(C2[i])) <= 2 * T::epsilon * std::abs(Y(C[i])));
    }
    std::cout << "OK: starpu::gemm::submit<" << T::short_name << "> restricted to CUDA\n";
}

template<typename T>
void validate_cuda_many()
{
    TransOp opT(TransOp::Trans), opN(TransOp::NoTrans);
    TransOp trans[2] = {opN, opT};
    Scalar alpha[3] = {0, 1, -3};
    Scalar beta[3] = {0, 1, 3};
    Index batch[3] = {1, 2, 100};
    for(auto transA: trans)
    {
        for(auto transB: trans)
        {
            for(Scalar a: alpha)
            {
                for(Scalar b: beta)
                {
                    for(auto nb: batch)
                    {
                        validate_cuda<T>(transA, transB, 10, 6, 3, nb, a, b);
                    }
                }
            }
        }
    }
}
#endif //NNTILE_USE_CUDA

int main(int argc, char **argv)
{
    // Initialize StarPU (it will automatically shutdown itself on exit)
    int ncpus=1, ncuda=1, ooc=0, verbose=0;
    const char *ooc_path = "/tmp/nntile_ooc";
    size_t ooc_size = 16777216;
    auto context = Context(ncpus, ncuda, ooc, ooc_path, ooc_size, verbose);

    // Launch all tests
#ifdef NNTILE_USE_CBLAS
    validate_cpu_many<fp32_t>();
    validate_cpu_many<fp64_t>();
#endif // NNTILE_USE_CBLAS
#ifdef NNTILE_USE_CUDA
    validate_cuda_many<fp32_t>();
    validate_cuda_many<fp64_t>();
#endif // NNTILE_USE_CUDA

    return 0;
}
