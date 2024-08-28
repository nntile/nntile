/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/starpu/sumprod_slice.cc
 * Scalar products of slices for two StarPU buffers
 *
 * @version 1.1.0
 * */

#include "nntile/starpu/sumprod_slice.hh"
#include "nntile/kernel/sumprod_slice.hh"
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
void validate_cpu(Index m, Index n, Index k, Scalar alpha, Scalar beta)
{
    using Y = typename T::repr_t;
    // Init all the data
    std::vector<T> src1(m*n*k), src2(src1);
    for(Index i = 0; i < m*n*k; ++i)
    {
        src1[i] = Y(i+1) / Y(i*i+1);
        src2[i] = Y(i*i+1);
    }
    std::vector<T> dst(m*n);
    for(Index i = 0; i < m*n; ++i)
    {
        dst[i] = Y(-i-1);
    }
    // Create copies of destination
    std::vector<T> dst2(dst);
    // Launch low-level kernel
    std::cout << "Run kernel::sumprod_slice::cpu<" << T::type_repr << ">\n";
    kernel::sumprod_slice::cpu<T>(m, n, k, alpha, &src1[0], &src2[0], beta,
            &dst[0]);
    // Check by actually submitting a task
    VariableHandle src1_handle(&src1[0], sizeof(T)*m*n*k, STARPU_R),
        src2_handle(&src2[0], sizeof(T)*m*n*k, STARPU_R),
        dst2_handle(&dst2[0], sizeof(T)*m*n, STARPU_RW);
    sumprod_slice::restrict_where(STARPU_CPU);
    std::cout << "Run starpu::sumprod_slice::submit<" << T::type_repr << "> restricted to CPU\n";
    sumprod_slice::submit<T>(m, n, k, alpha, src1_handle, src2_handle, beta,
            dst2_handle);
    starpu_task_wait_for_all();
    dst2_handle.unregister();
    // Check result
    for(Index i = 0; i < m*n; ++i)
    {
        TEST_ASSERT(Y(dst[i]) == Y(dst2[i]));
    }
    std::cout << "OK: starpu::sumprod_slice::submit<" << T::type_repr << "> restricted to CPU\n";
}

#ifdef NNTILE_USE_CUDA
template<typename T>
void validate_cuda(Index m, Index n, Index k, Scalar alpha, Scalar beta)
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
    std::vector<T> src1(m*n*k), src2(src1);
    for(Index i = 0; i < m*n*k; ++i)
    {
        src1[i] = Y(i+1) / Y(i*i+1);
        src2[i] = Y(i*i+1);
    }
    std::vector<T> dst(m*n);
    for(Index i = 0; i < m*n; ++i)
    {
        dst[i] = Y(-i-1);
    }
    // Create copies of destination
    std::vector<T> dst2(dst);
    // Launch low-level kernel
    T *dev_src1, *dev_src2, *dev_dst;
    cuda_err = cudaMalloc(&dev_src1, sizeof(T)*m*n*k);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMalloc(&dev_src2, sizeof(T)*m*n*k);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMalloc(&dev_dst, sizeof(T)*m*n);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_src1, &src1[0], sizeof(T)*m*n*k,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_src2, &src2[0], sizeof(T)*m*n*k,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_dst, &dst[0], sizeof(T)*m*n,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    std::cout << "Run kernel::sumprod_slice::cuda<" << T::type_repr << ">\n";
    kernel::sumprod_slice::cuda<T>(stream, m, n, k, alpha, dev_src1, dev_src2,
            beta, dev_dst);
    // Wait for result and destroy stream
    cuda_err = cudaStreamSynchronize(stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaStreamDestroy(stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Copy result back to CPU
    cuda_err = cudaMemcpy(&dst[0], dev_dst, sizeof(T)*m*n,
            cudaMemcpyDeviceToHost);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Deallocate CUDA memory
    cuda_err = cudaFree(dev_src1);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaFree(dev_src2);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaFree(dev_dst);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Check by actually submitting a task
    VariableHandle src1_handle(&src1[0], sizeof(T)*m*n*k, STARPU_R),
        src2_handle(&src2[0], sizeof(T)*m*n*k, STARPU_R),
        dst2_handle(&dst2[0], sizeof(T)*m*n, STARPU_RW);
    sumprod_slice::restrict_where(STARPU_CUDA);
    std::cout << "Run starpu::sumprod_slice::submit<" << T::type_repr << "> restricted to CUDA\n";
    sumprod_slice::submit<T>(m, n, k, alpha, src1_handle, src2_handle, beta,
            dst2_handle);
    starpu_task_wait_for_all();
    dst2_handle.unregister();
    // Check result
    for(Index i = 0; i < m*n; ++i)
    {
        TEST_ASSERT(Y(dst[i]) == Y(dst2[i]));
    }
    std::cout << "OK: starpu::sumprod_slice::submit<" << T::type_repr << "> restricted to CUDA\n";
}
#endif // NNTILE_USE_CUDA

int main(int argc, char **argv)
{
    // Init StarPU for testing
    Config starpu(1, 1, 0);
    // Init codelet
    sumprod_slice::init();
    // Launch all tests
    validate_cpu<fp32_t>(3, 5, 7, 2.0, -1.0);
    validate_cpu<fp32_t>(3, 5, 7, -1.0, 0.0);
    validate_cpu<fp64_t>(3, 5, 7, 2.0, -1.0);
    validate_cpu<fp64_t>(3, 5, 7, -1.0, 0.0);
#ifdef NNTILE_USE_CUDA
    validate_cuda<fp32_t>(3, 5, 7, 2.0, -1.0);
    validate_cuda<fp32_t>(3, 5, 7, -1.0, 0.0);
    validate_cuda<fp64_t>(3, 5, 7, 2.0, -1.0);
    validate_cuda<fp64_t>(3, 5, 7, -1.0, 0.0);
#endif // NNTILE_USE_CUDA
    return 0;
}
