/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/starpu/normalize.cc
 * Normalize operation on a StarPU buffer
 *
 * @version 1.1.0
 * */

#include "nntile/starpu/normalize.hh"
#include "nntile/kernel/normalize.hh"
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
void validate_cpu(Index m, Index n, Index k, Index l, Scalar eps, T gamma, T beta)
{
    using Y = typename T::repr_t;
    // Init all the data
    std::vector<T> sumnorm(2*m*n);
    for(Index i = 0; i < 2*m*n; i += 2)
    {
        sumnorm[i] = Y(l*i); // Sum
        sumnorm[i+1] = std::sqrt(Y(l)*i*i + Y(l)); // Norm
    }
    std::vector<T> dst(m*n*k);
    for(Index i = 0; i < m*n*k; ++i)
    {
        dst[i] = Y(-i-1);
    }
    // Create copies of destination
    std::vector<T> dst2(dst);
    // Launch low-level kernel
    std::cout << "Run kernel::normalize::cpu<" << T::type_repr << ">\n";
    kernel::normalize::cpu<T>(m, n, k, l, eps, &gamma, &beta, &sumnorm[0],
            &dst[0]);
    // Check by actually submitting a task
    T gamma_beta[2] = {gamma, beta};
    VariableHandle sumnorm_handle(&sumnorm[0], sizeof(T)*2*m*n,
            STARPU_R),
        dst2_handle(&dst2[0], sizeof(T)*m*n*k, STARPU_RW),
        gamma_beta_handle(gamma_beta, sizeof(gamma_beta), STARPU_R);
    normalize::restrict_where(STARPU_CPU);
    std::cout << "Run starpu::normalize::submit<" << T::type_repr << "> restricted to CPU\n";
    normalize::submit<T>(m, n, k, l, eps, gamma_beta_handle, sumnorm_handle,
            dst2_handle);
    starpu_task_wait_for_all();
    dst2_handle.unregister();
    // Check result
    for(Index i = 0; i < m*n*k; ++i)
    {
        TEST_ASSERT(Y(dst[i]) == Y(dst2[i]));
    }
    std::cout << "OK: starpu::normalize::submit<" << T::type_repr << "> restricted to CPU\n";
}

template<typename T>
void validate_many_cpu()
{
    validate_cpu<T>(3, 5, 7, 10, 1e-16, T(1.0), T(0.0));
    validate_cpu<T>(3, 5, 7, 2, 10, T(2.0), T(1.0));
}

#ifdef NNTILE_USE_CUDA
template<typename T>
void validate_cuda(Index m, Index n, Index k, Index l, Scalar eps, T gamma, T beta)
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
    std::vector<T> sumnorm(2*m*n);
    for(Index i = 0; i < 2*m*n; i += 2)
    {
        sumnorm[i] = Y(l*i); // Sum
        sumnorm[i+1] = std::sqrt(Y(l)*i*i + Y(l)); // Norm
    }
    std::vector<T> dst(m*n*k);
    for(Index i = 0; i < m*n*k; ++i)
    {
        dst[i] = Y(-i-1);
    }
    // Create copies of destination
    std::vector<T> dst2(dst);
    // Launch low-level kernel
    T *dev_sumnorm, *dev_dst, *dev_gamma, *dev_beta;
    cuda_err = cudaMalloc(&dev_sumnorm, sizeof(T)*2*m*n);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMalloc(&dev_dst, sizeof(T)*m*n*k);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMalloc(&dev_gamma, sizeof(T));
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMalloc(&dev_beta, sizeof(T));
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_sumnorm, &sumnorm[0], sizeof(T)*2*m*n,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_dst, &dst[0], sizeof(T)*m*n*k,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_gamma, &gamma, sizeof(T),
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_beta, &beta, sizeof(T),
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    std::cout << "Run kernel::normalize::cuda<" << T::type_repr << ">\n";
    kernel::normalize::cuda<T>(stream, m, n, k, l, eps, dev_gamma, dev_beta,
            dev_sumnorm, dev_dst);
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
    cuda_err = cudaFree(dev_sumnorm);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaFree(dev_dst);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaFree(dev_gamma);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaFree(dev_beta);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Check by actually submitting a task
    T gamma_beta[2] = {gamma, beta};
    VariableHandle sumnorm_handle(&sumnorm[0], sizeof(T)*2*m*n,
            STARPU_R),
        dst2_handle(&dst2[0], sizeof(T)*m*n*k, STARPU_RW),
        gamma_beta_handle(gamma_beta, sizeof(T)*2, STARPU_R);
    normalize::restrict_where(STARPU_CUDA);
    std::cout << "Run starpu::normalize::submit<" << T::type_repr << "> restricted to CUDA\n";
    normalize::submit<T>(m, n, k, l, eps, gamma_beta_handle, sumnorm_handle,
            dst2_handle);
    starpu_task_wait_for_all();
    dst2_handle.unregister();
    // Check result
    for(Index i = 0; i < m*n*k; ++i)
    {
        TEST_ASSERT(Y(dst[i]) == Y(dst2[i]));
    }
    std::cout << "OK: starpu::normalize::submit<" << T::type_repr << "> restricted to CUDA\n";
}

template<typename T>
void validate_many_cuda()
{
    validate_cuda<T>(3, 5, 7, 10, 1e-16, T(1.0), T(0.0));
    validate_cuda<T>(3, 5, 7, 2, 10, T(2.0), T(1.0));
}
#endif // NNTILE_USE_CUDA

int main(int argc, char **argv)
{
    // Init StarPU for testing
    Config starpu(1, 1, 0);
    // Init codelet
    normalize::init();
    // Launch all tests
    validate_many_cpu<fp32_t>();
    validate_many_cpu<fp64_t>();
#ifdef NNTILE_USE_CUDA
    validate_many_cuda<fp32_t>();
    validate_many_cuda<fp64_t>();
#endif // NNTILE_USE_CUDA
    return 0;
}
