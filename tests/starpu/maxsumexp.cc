/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/starpu/maxsumexp.cc
 * Max and sum of exponents for StarPU buffer
 *
 * @version 1.1.0
 * */

#include "nntile/starpu/maxsumexp.hh"
#include "nntile/kernel/maxsumexp.hh"
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
    // Init all the data
    std::vector<T> src(m*n*k);
    for(Index i = 0; i < m*n*k; ++i)
    {
        src[i] = Y(i+1);
    }
    std::vector<T> maxsumexp(2*m*n);
    for(Index i = 0; i < 2*m*n; i += 2)
    {
        maxsumexp[i] = Y(-i-1); // Max
        maxsumexp[i+1] = Y(2*i); // Sum of exponents
    }
    // Create copies of destination
    std::vector<T> maxsumexp2(maxsumexp);
    // Launch low-level kernel
    std::cout << "Run kernel::maxsumexp::cpu<" << T::type_repr << ">\n";
    kernel::maxsumexp::cpu<T>(m, n, k, &src[0], &maxsumexp[0]);
    // Check by actually submitting a task
    VariableHandle src_handle(&src[0], sizeof(T)*m*n*k, STARPU_R),
        maxsumexp2_handle(&maxsumexp2[0], sizeof(T)*2*m*n, STARPU_RW);
    maxsumexp::restrict_where(STARPU_CPU);
    std::cout << "Run starpu::maxsumexp::submit<" << T::type_repr << "> restricted to CPU\n";
    maxsumexp::submit<T>(m, n, k, src_handle, maxsumexp2_handle);
    starpu_task_wait_for_all();
    maxsumexp2_handle.unregister();
    // Check result
    for(Index i = 0; i < m*n; ++i)
    {
        TEST_ASSERT(Y(maxsumexp[i]) == Y(maxsumexp2[i]));
    }
    std::cout << "OK: starpu::maxsumexp::submit<" << T::type_repr << "> restricted to CPU\n";
}

#ifdef NNTILE_USE_CUDA
template<typename T>
void validate_cuda(Index m, Index n, Index k)
{
    using Y = typename T::repr_t;
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
    std::vector<T> src(m*n*k);
    for(Index i = 0; i < m*n*k; ++i)
    {
        src[i] = Y(i+1);
    }
    std::vector<T> maxsumexp(2*m*n);
    for(Index i = 0; i < 2*m*n; i += 2)
    {
        maxsumexp[i] = Y(-i-1); // Max
        maxsumexp[i+1] = Y(2*i); // Sum of exponents
    }
    // Create copies of destination
    std::vector<T> maxsumexp2(maxsumexp);
    // Launch low-level kernel
    T *dev_src, *dev_maxsumexp;
    cuda_err = cudaMalloc(&dev_src, sizeof(T)*m*n*k);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMalloc(&dev_maxsumexp, sizeof(T)*2*m*n);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_src, &src[0], sizeof(T)*m*n*k,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_maxsumexp, &maxsumexp[0], sizeof(T)*2*m*n,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    std::cout << "Run kernel::maxsumexp::cuda<" << T::type_repr << ">\n";
    kernel::maxsumexp::cuda<T>(stream, m, n, k, dev_src, dev_maxsumexp);
    // Wait for result and destroy stream
    cuda_err = cudaStreamSynchronize(stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaStreamDestroy(stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Copy result back to CPU
    cuda_err = cudaMemcpy(&maxsumexp[0], dev_maxsumexp, sizeof(T)*2*m*n,
            cudaMemcpyDeviceToHost);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Deallocate CUDA memory
    cuda_err = cudaFree(dev_src);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaFree(dev_maxsumexp);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Check by actually submitting a task
    VariableHandle src_handle(&src[0], sizeof(T)*m*n*k, STARPU_R),
        maxsumexp2_handle(&maxsumexp2[0], sizeof(T)*2*m*n, STARPU_RW);
    maxsumexp::restrict_where(STARPU_CUDA);
    std::cout << "Run starpu::maxsumexp::submit<" << T::type_repr << "> restricted to CUDA\n";
    maxsumexp::submit<T>(m, n, k, src_handle, maxsumexp2_handle);
    starpu_task_wait_for_all();
    maxsumexp2_handle.unregister();
    // Check result
    for(Index i = 0; i < 2*m*n; ++i)
    {
        TEST_ASSERT(Y(maxsumexp[i]) == Y(maxsumexp2[i]));
    }
    std::cout << "OK: starpu::maxsumexp::submit<" << T::type_repr << "> restricted to CUDA\n";
}
#endif // NNTILE_USE_CUDA

int main(int argc, char **argv)
{
    // Init StarPU for testing
    Config starpu(1, 1, 0);
    // Init codelet
    maxsumexp::init();
    // Launch all tests
    validate_cpu<fp32_t>(3, 5, 7);
    validate_cpu<fp64_t>(3, 5, 7);
#ifdef NNTILE_USE_CUDA
    validate_cuda<fp32_t>(3, 5, 7);
    validate_cuda<fp64_t>(3, 5, 7);
#endif // NNTILE_USE_CUDA
    return 0;
}
