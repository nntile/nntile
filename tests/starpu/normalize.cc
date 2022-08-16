/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/starpu/normalize.cc
 * Normalize operation on a StarPU buffer
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-16
 * */

#include "nntile/starpu/normalize.hh"
#include "nntile/kernel/cpu/normalize.hh"
#ifdef NNTILE_USE_CUDA
#   include "nntile/kernel/cuda/normalize.hh"
#   include <cuda_runtime.h>
#endif // NNTILE_USE_CUDA
#include <vector>
#include <stdexcept>
#include <cmath>
#include <iostream>

using namespace nntile;

template<typename T>
void validate_cpu(Index m, Index n, Index k, Index l, T eps, T gamma, T beta)
{
    // Init all the data
    std::vector<T> src(2*m*n);
    for(Index i = 0; i < 2*m*n; i += 2)
    {
        src[i] = T(l*i); // Sum
        src[i+1] = std::sqrt(T(l)*i*i + T(l)); // Norm
    }
    std::vector<T> dst(m*n*k);
    for(Index i = 0; i < m*n*k; ++i)
    {
        dst[i] = T(-i-1);
    }
    // Create copies of destination
    std::vector<T> dst2(dst);
    // Launch low-level kernel
    std::cout << "Run cpu::normalize<T>\n";
    kernel::cpu::normalize<T>(m, n, k, l, eps, gamma, beta, &src[0], &dst[0]);
    // Check by actually submitting a task
    T gamma_beta[2] = {gamma, beta};
    StarpuVariableHandle src_handle(&src[0], sizeof(T)*2*m*n, STARPU_R),
        dst2_handle(&dst2[0], sizeof(T)*m*n*k, STARPU_RW),
        gamma_beta_handle(gamma_beta, sizeof(gamma_beta), STARPU_R);
    starpu::normalize_restrict_where(STARPU_CPU);
    starpu_resume();
    std::cout << "Run starpu::normalize<T> restricted to CPU\n";
    starpu::normalize<T>(m, n, k, l, eps, gamma_beta_handle, src_handle,
            dst2_handle);
    starpu_task_wait_for_all();
    dst2_handle.unregister();
    starpu_pause();
    // Check result
    for(Index i = 0; i < m*n*k; ++i)
    {
        if(dst[i] != dst2[i])
        {
            throw std::runtime_error("StarPU submission wrong result");
        }
    }
    std::cout << "OK: starpu::normalize<T> restricted to CPU\n";
}

template<typename T>
void validate_many_cpu()
{
    validate_cpu<T>(3, 5, 7, 10, 0, 1, 0);
    validate_cpu<T>(3, 5, 7, 2, 10, 2, 1);
}

#ifdef NNTILE_USE_CUDA
template<typename T>
void validate_cuda(Index m, Index n, Index k, Index l, T eps, T gamma, T beta)
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
    std::vector<T> src(2*m*n);
    for(Index i = 0; i < 2*m*n; i += 2)
    {
        src[i] = T(l*i); // Sum
        src[i+1] = std::sqrt(T(l)*i*i + T(l)); // Norm
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
    cuda_err = cudaMalloc(&dev_src, sizeof(T)*2*m*n);
    if(cuda_err != cudaSuccess)
    {
        throw std::runtime_error("CUDA error");
    }
    cuda_err = cudaMalloc(&dev_dst, sizeof(T)*m*n*k);
    if(cuda_err != cudaSuccess)
    {
        throw std::runtime_error("CUDA error");
    }
    cuda_err = cudaMemcpy(dev_src, &src[0], sizeof(T)*2*m*n,
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
    std::cout << "Run cuda::normalize<T>\n";
    kernel::cuda::normalize<T>(stream, m, n, k, l, eps, gamma, beta, dev_src,
            dev_dst);
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
    T gamma_beta[2] = {gamma, beta};
    std::cout << "sizeof(gamma_beta)=" << sizeof(gamma_beta) << "\n";
    StarpuVariableHandle src_handle(&src[0], sizeof(T)*2*m*n, STARPU_R),
        dst2_handle(&dst2[0], sizeof(T)*m*n*k, STARPU_RW),
        gamma_beta_handle(gamma_beta, sizeof(T)*2, STARPU_R);
    starpu::normalize_restrict_where(STARPU_CUDA);
    starpu_resume();
    std::cout << "Run starpu::normalize<T> restricted to CUDA\n";
    starpu::normalize<T>(m, n, k, l, eps, gamma_beta_handle, src_handle,
            dst2_handle);
    starpu_task_wait_for_all();
    dst2_handle.unregister();
    starpu_pause();
    return;
    // Check result
    for(Index i = 0; i < m*n*k; ++i)
    {
        if(dst[i] != dst2[i])
        {
            throw std::runtime_error("StarPU submission wrong result");
        }
    }
    std::cout << "OK: starpu::normalize<T> restricted to CUDA\n";
}

template<typename T>
void validate_many_cuda()
{
    validate_cuda<T>(3, 5, 7, 10, 0, 1, 0);
    validate_cuda<T>(3, 5, 7, 2, 10, 2, 1);
}
#endif // NNTILE_USE_CUDA

int main(int argc, char **argv)
{
    // Init StarPU configuration and set number of CPU workers to 1
    starpu_conf conf;
    int ret = starpu_conf_init(&conf);
    if(ret != 0)
    {
        throw std::runtime_error("starpu_conf_init error");
    }
    conf.ncpus = 1;
#ifdef NNTILE_USE_CUDA
    conf.ncuda = 1;
#else // NNTILE_USE_CUDA
    conf.ncuda = 0;
#endif // NNTILE_USE_CUDA
    ret = starpu_init(&conf);
    if(ret != 0)
    {
        throw std::runtime_error("starpu_init error");
    }
    // Launch all tests
    starpu_pause();
    validate_many_cpu<fp32_t>();
    validate_many_cpu<fp64_t>();
#ifdef NNTILE_USE_CUDA
    validate_many_cuda<fp32_t>();
    validate_many_cuda<fp64_t>();
#endif // NNTILE_USE_CUDA
    starpu_resume();
    starpu_shutdown();
    return 0;
}

