/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/starpu/softmax_inplace.cc
 * softmax_inplace operation on a StarPU buffer
 *
 * @version 1.1.0
 * */

#include "nntile/starpu/softmax_inplace.hh"
#include "nntile/kernel/softmax_inplace.hh"
#include "../testing.hh"
#ifdef NNTILE_USE_CUDA
#   include <cuda_runtime.h>
#endif // NNTILE_USE_CUDA
#include <vector>
#include <stdexcept>
#include <cmath>
#include <iostream>

using namespace nntile;
using namespace nntile::starpu;

template<typename T>
void validate_cpu(Index m, Index n, Index k)
{
    using Y = typename T::repr_t;
    constexpr Scalar alpha = 1.0;
    // Init all the data
    std::vector<T> maxsumexp(2*m*n);
    for(Index i = 0; i < 2*m*n; i += 2)
    {
        maxsumexp[i] = Y(i); // Max
        maxsumexp[i+1] = std::exp(Y(i) / Y{10}); // Sum of exponents
    }
    std::vector<T> dst(m*n*k);
    for(Index i = 0; i < m*n*k; ++i)
    {
        dst[i] = Y(-i-1);
    }
    // Create copies of destination
    std::vector<T> dst2(dst);
    // Launch low-level kernel
    std::cout << "Run kernel::softmax_inplace::cpu<" << T::type_repr << ">\n";
    kernel::softmax_inplace::cpu<T>(m, n, k, &maxsumexp[0], alpha, &dst[0]);
    // Check by actually submitting a task
    VariableHandle maxsumexp_handle(&maxsumexp[0], sizeof(T)*2*m*n,
            STARPU_R),
        dst2_handle(&dst2[0], sizeof(T)*m*n*k, STARPU_RW);
    softmax_inplace::restrict_where(STARPU_CPU);
    std::cout << "Run starpu::softmax_inplace::submit<" << T::type_repr << "> restricted to CPU\n";
    softmax_inplace::submit<T>(m, n, k, maxsumexp_handle, alpha, dst2_handle);
    starpu_task_wait_for_all();
    dst2_handle.unregister();
    // Check result
    for(Index i = 0; i < m*n*k; ++i)
    {
        TEST_ASSERT(Y(dst[i]) == Y(dst2[i]));
    }
    std::cout << "OK: starpu::softmax_inplace::submit<" << T::type_repr << "> restricted to CPU\n";
}

#ifdef NNTILE_USE_CUDA
template<typename T>
void validate_cuda(Index m, Index n, Index k)
{
    using Y = typename T::repr_t;
    constexpr Scalar alpha = 1.0;
    // Get a StarPU CUDA worker (to perform computations on the same device)
    int cuda_worker_id = starpu_worker_get_by_type(STARPU_CUDA_WORKER, 0);
    // Choose worker CUDA device
    int dev_id = starpu_worker_get_devid(cuda_worker_id);
    cudaError_t cuda_err = cudaSetDevice(dev_id);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Create CUDA stream
    cudaStream_t stream;
    cuda_err = cudaStreamCreate(&stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Init all the data
    std::vector<T> maxsumexp(2*m*n);
    for(Index i = 0; i < 2*m*n; i += 2)
    {
        maxsumexp[i] = Y(i); // Max
        maxsumexp[i+1] = std::exp(Y(i) / Y{10}); // Sum of exponents
    }
    std::vector<T> dst(m*n*k);
    for(Index i = 0; i < m*n*k; ++i)
    {
        dst[i] = Y(-i-1);
    }
    // Create copies of destination
    std::vector<T> dst2(dst);
    // Launch low-level kernel
    T *dev_maxsumexp, *dev_dst;
    cuda_err = cudaMalloc(&dev_maxsumexp, sizeof(T)*2*m*n);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMalloc(&dev_dst, sizeof(T)*m*n*k);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_maxsumexp, &maxsumexp[0], sizeof(T)*2*m*n,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_dst, &dst[0], sizeof(T)*m*n*k,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    std::cout << "Run kernel::softmax_inplace::cuda<" << T::type_repr << ">\n";
    kernel::softmax_inplace::cuda<T>(stream, m, n, k, dev_maxsumexp, alpha,
            dev_dst);
    // Wait for result and destroy stream
    cuda_err = cudaStreamSynchronize(stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaStreamDestroy(stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Copy result back to CPU
    cuda_err = cudaMemcpy(&dst[0], dev_dst, sizeof(T)*m*n*k,
            cudaMemcpyDeviceToHost);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Deallocate CUDA memory
    cuda_err = cudaFree(dev_maxsumexp);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaFree(dev_dst);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Check by actually submitting a task
    VariableHandle maxsumexp_handle(&maxsumexp[0], sizeof(T)*2*m*n,
            STARPU_R),
        dst2_handle(&dst2[0], sizeof(T)*m*n*k, STARPU_RW);
    softmax_inplace::restrict_where(STARPU_CUDA);
    std::cout << "Run starpu::softmax_inplace::submit<" << T::type_repr << "> restricted to CUDA\n";
    softmax_inplace::submit<T>(m, n, k, maxsumexp_handle, alpha, dst2_handle);
    starpu_task_wait_for_all();
    dst2_handle.unregister();
    // Check result
    for(Index i = 0; i < m*n*k; ++i)
    {
        TEST_ASSERT(Y(dst[i]) == Y(dst2[i]));
    }
    std::cout << "OK: starpu::softmax_inplace::submit<" << T::type_repr << "> restricted to CUDA\n";
}
#endif // NNTILE_USE_CUDA

int main(int argc, char **argv)
{
    // Init StarPU for testing
    Config starpu(1, 1, 0);
    // Init codelet
    softmax_inplace::init();
    // Launch all tests
    validate_cpu<fp32_t>(3, 5, 7);
    validate_cpu<fp64_t>(3, 5, 7);
#ifdef NNTILE_USE_CUDA
    validate_cuda<fp32_t>(3, 5, 7);
    validate_cuda<fp64_t>(3, 5, 7);
#endif // NNTILE_USE_CUDA
    return 0;
}
