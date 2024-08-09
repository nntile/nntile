/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/starpu/transpose.cc
 * Transpose operation
 *
 * @version 1.1.0
 * */

#include "nntile/starpu/transpose.hh"
#include "nntile/kernel/transpose.hh"
#include "../testing.hh"
#ifdef NNTILE_USE_CUDA
#   include <cuda_runtime.h>
#endif // NNTILE_USE_CUDA
#include <vector>
#include <stdexcept>
#include <iostream>

using namespace nntile;
using namespace nntile::starpu;

template<typename T>
void validate_cpu(Index m, Index n)
{
    using Y = typename T::repr_t;
    // Init all the data
    std::vector<T> src(m*n);
    for(Index i = 0; i < m*n; ++i)
    {
        src[i] = Y(2*i+2);
    }
    std::vector<T> dst(m*n);
    for(Index i = 0; i < m*n; ++i)
    {
        dst[i] = Y(-i-1);
    }
    // Create copies of destination
    std::vector<T> dst2(dst);
    // Launch low-level kernel
    std::cout << "Run kernel::transpose::cpu<" << T::type_repr << ">\n";
    kernel::transpose::cpu<T>(m, n, 0.5, &src[0], &dst[0]);
    // Check by actually submitting a task
    VariableHandle src_handle(&src[0], sizeof(T)*m*n, STARPU_R),
        dst2_handle(&dst2[0], sizeof(T)*m*n, STARPU_RW);
    transpose::restrict_where(STARPU_CPU);
    std::cout << "Run starpu::transpose::submit<" << T::type_repr << "> restricted to CPU\n";
    transpose::submit<T>(m, n, 0.5, src_handle, dst2_handle);
    starpu_task_wait_for_all();
    dst2_handle.unregister();
    // Check result
    for(Index i = 0; i < m*n; ++i)
    {
        TEST_ASSERT(Y(dst[i]) == Y(dst2[i]));
    }
    std::cout << "OK: starpu::transpose::submit<" << T::type_repr << "> restricted to CPU\n";
}

#ifdef NNTILE_USE_CUDA
template<typename T>
void validate_cuda(Index m, Index n)
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
    std::vector<T> src(m*n);
    for(Index i = 0; i < m*n; ++i)
    {
        src[i] = Y(2*i+2);
    }
    std::vector<T> dst(m*n);
    for(Index i = 0; i < m*n; ++i)
    {
        dst[i] = Y(-i-1);
    }
    // Create copies of destination
    std::vector<T> dst2(dst);
    // Launch low-level kernel
    T *dev_src, *dev_dst;
    cuda_err = cudaMalloc(&dev_src, sizeof(T)*m*n);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMalloc(&dev_dst, sizeof(T)*m*n);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_src, &src[0], sizeof(T)*m*n,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_dst, &dst[0], sizeof(T)*m*n,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    std::cout << "Run kernel::transpose::cuda<" << T::type_repr << ">\n";
    kernel::transpose::cuda<T>(stream, m, n, 0.5, dev_src, dev_dst);
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
    cuda_err = cudaFree(dev_src);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaFree(dev_dst);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Check by actually submitting a task
    VariableHandle src_handle(&src[0], sizeof(T)*m*n, STARPU_R),
        dst2_handle(&dst2[0], sizeof(T)*m*n, STARPU_RW);
    transpose::restrict_where(STARPU_CUDA);
    std::cout << "Run starpu::transpose::submit<" << T::type_repr << "> restricted to CUDA\n";
    transpose::submit<T>(m, n, 0.5, src_handle, dst2_handle);
    starpu_task_wait_for_all();
    dst2_handle.unregister();
    // Check result
    for(Index i = 0; i < m*n; ++i)
    {
        TEST_ASSERT(Y(dst[i]) == Y(dst2[i]));
    }
    std::cout << "OK: starpu::transpose::submit<" << T::type_repr << "> restricted to CUDA\n";
}
#endif // NNTILE_USE_CUDA

int main(int argc, char **argv)
{
    // Init StarPU for testing
    Config starpu(1, 1, 0);
    // Init codelet
    transpose::init();
    // Launch all tests
    // Bias for middle axis
    validate_cpu<fp32_t>(3, 5);
    validate_cpu<fp64_t>(3, 5);
#ifdef NNTILE_USE_CUDA
    validate_cuda<fp32_t>(3, 5);
    validate_cuda<fp64_t>(3, 5);
#endif // NNTILE_USE_CUDA
    return 0;
}
