/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/starpu/bias.cc
 * Bias operation on a StarPU buffer
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-09-06
 * */

#include "nntile/starpu/bias.hh"
#include "nntile/kernel/bias/cpu.hh"
#ifdef NNTILE_USE_CUDA
#   include "nntile/kernel/bias/cuda.hh"
#   include <cuda_runtime.h>
#endif // NNTILE_USE_CUDA
#include "common.hh"
#include <vector>
#include <stdexcept>
#include <iostream>

using namespace nntile;

template<typename T>
void validate_cpu(Index m, Index n, Index k)
{
    // Init all the data
    std::vector<T> src(m*n);
    for(Index i = 0; i < m*n; ++i)
    {
        src[i] = T(i+1);
    }
    std::vector<T> dst(m*n*k);
    for(Index i = 0; i < m*n*k; ++i)
    {
        dst[i] = T(-i-1);
    }
    // Create copies of destination
    std::vector<T> dst2(dst);
    // Launch low-level kernel
    std::cout << "Run kernel::bias::cpu<T>\n";
    kernel::bias::cpu<T>(m, n, k, &src[0], &dst[0]);
    // Check by actually submitting a task
    StarpuVariableHandle src_handle(&src[0], sizeof(T)*m*n, STARPU_R),
        dst2_handle(&dst2[0], sizeof(T)*m*n*k, STARPU_RW);
    starpu::bias::restrict_where(STARPU_CPU);
    std::cout << "Run starpu::bias::submit<T> restricted to CPU\n";
    starpu::bias::submit<T>(m, n, k, src_handle, dst2_handle);
    starpu_task_wait_for_all();
    dst2_handle.unregister();
    // Check result
    for(Index i = 0; i < m*n*k; ++i)
    {
        if(dst[i] != dst2[i])
        {
            throw std::runtime_error("StarPU submission wrong result");
        }
    }
    std::cout << "OK: starpu::bias::submit<T> restricted to CPU\n";
}

#ifdef NNTILE_USE_CUDA
template<typename T>
void validate_cuda(Index m, Index n, Index k)
{
    // Get a StarPU CUDA worker (to perform computations on the same device)
    int cuda_worker_id = starpu_worker_get_by_type(STARPU_CUDA_WORKER, 0);
    // Choose worker CUDA device
    int dev_id = starpu_worker_get_devid(cuda_worker_id);
    cudaError_t cuda_err = cudaSetDevice(dev_id);
    if(cuda_err != cudaSuccess)
    {
        throw std::runtime_error("CUDA error");
    }
    // Create CUDA stream
    cudaStream_t stream;
    cuda_err = cudaStreamCreate(&stream);
    if(cuda_err != cudaSuccess)
    {
        throw std::runtime_error("CUDA error");
    }
    // Init all the data
    std::vector<T> src(m*n);
    for(Index i = 0; i < m*n; ++i)
    {
        src[i] = T(i+1);
    }
    std::vector<T> dst(m*n*k);
    for(Index i = 0; i < m*n*k; ++i)
    {
        dst[i] = T(-i-1);
    }
    // Create copies of destination
    std::vector<T> dst2(dst);
    // Launch low-level kernel
    T *dev_src, *dev_dst;
    cuda_err = cudaMalloc(&dev_src, sizeof(T)*m*n);
    if(cuda_err != cudaSuccess)
    {
        throw std::runtime_error("CUDA error");
    }
    cuda_err = cudaMalloc(&dev_dst, sizeof(T)*m*n*k);
    if(cuda_err != cudaSuccess)
    {
        throw std::runtime_error("CUDA error");
    }
    cuda_err = cudaMemcpy(dev_src, &src[0], sizeof(T)*m*n,
            cudaMemcpyHostToDevice);
    if(cuda_err != cudaSuccess)
    {
        throw std::runtime_error("CUDA error");
    }
    cuda_err = cudaMemcpy(dev_dst, &dst[0], sizeof(T)*m*n*k,
            cudaMemcpyHostToDevice);
    if(cuda_err != cudaSuccess)
    {
        throw std::runtime_error("CUDA error");
    }
    std::cout << "Run kernel::bias::cuda<T>\n";
    kernel::bias::cuda<T>(stream, m, n, k, dev_src, dev_dst);
    // Wait for result and destroy stream
    cuda_err = cudaStreamSynchronize(stream);
    if(cuda_err != cudaSuccess)
    {
        throw std::runtime_error("CUDA error");
    }
    cuda_err = cudaStreamDestroy(stream);
    if(cuda_err != cudaSuccess)
    {
        throw std::runtime_error("CUDA error");
    }
    // Copy result back to CPU
    cuda_err = cudaMemcpy(&dst[0], dev_dst, sizeof(T)*m*n*k,
            cudaMemcpyDeviceToHost);
    if(cuda_err != cudaSuccess)
    {
        throw std::runtime_error("CUDA error");
    }
    // Deallocate CUDA memory
    cuda_err = cudaFree(dev_src);
    if(cuda_err != cudaSuccess)
    {
        throw std::runtime_error("CUDA error");
    }
    cuda_err = cudaFree(dev_dst);
    if(cuda_err != cudaSuccess)
    {
        throw std::runtime_error("CUDA error");
    }
    // Check by actually submitting a task
    StarpuVariableHandle src_handle(&src[0], sizeof(T)*m*n, STARPU_R),
        dst2_handle(&dst2[0], sizeof(T)*m*n*k, STARPU_RW);
    starpu::bias::restrict_where(STARPU_CUDA);
    std::cout << "Run starpu::bias::submit<T> restricted to CUDA\n";
    starpu::bias::submit<T>(m, n, k, src_handle, dst2_handle);
    starpu_task_wait_for_all();
    dst2_handle.unregister();
    // Check result
    for(Index i = 0; i < m*n*k; ++i)
    {
        if(dst[i] != dst2[i])
        {
            throw std::runtime_error("StarPU submission wrong result");
        }
    }
    std::cout << "OK: starpu::bias::submit<T> restricted to CUDA\n";
}
#endif // NNTILE_USE_CUDA

int main(int argc, char **argv)
{
    // Init StarPU for testing
    StarpuTest starpu;
    // Init codelet
    starpu::bias::init();
    // Launch all tests
    validate_cpu<fp32_t>(3, 5, 7);
    validate_cpu<fp64_t>(3, 5, 7);
#ifdef NNTILE_USE_CUDA
    validate_cuda<fp32_t>(3, 5, 7);
    validate_cuda<fp64_t>(3, 5, 7);
#endif // NNTILE_USE_CUDA
    return 0;
}

