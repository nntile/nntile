/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/add_fiber.cc
 * Per-element addition of a tensor and a broadcasted fiber
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/add_fiber.hh"
#include "../testing.hh"
#include <vector>
#include <stdexcept>
#include <limits>
#include <iostream>
#include <random>
#include "nntile/kernel/cpu.hh"
#include "nntile/kernel/cuda.hh"

#ifdef NNTILE_USE_CUDA
#include <cuda_runtime.h>
#endif // NNTILE_USE_CUDA

using namespace nntile;
using namespace nntile::kernel;

#ifdef NNTILE_USE_CUDA
template<typename T>
void run_cuda(Index m, Index n, Index k, Index batch, Scalar alpha, const std::vector<T> &src1,
        Scalar beta, const std::vector<T> &src2, std::vector<T> &dst)
{
    // Copy to device
    T *dev_src1, *dev_src2, *dev_dst;
    cudaError_t cuda_err = cudaMalloc(&dev_src1, sizeof(T)*k*batch);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMalloc(&dev_src2, sizeof(T)*m*n*k*batch);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMalloc(&dev_dst, sizeof(T)*m*n*k*batch);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_src1, &src1[0], sizeof(T)*k*batch,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_src2, &src2[0], sizeof(T)*m*n*k*batch,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_dst, &dst[0], sizeof(T)*m*n*k*batch,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Init stream
    cudaStream_t stream;
    cuda_err = cudaStreamCreate(&stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Launch low-level CUDA kernel
    add_fiber::cuda<T>(stream, m, n, k, batch, alpha, dev_src1, beta, dev_src2, dev_dst);
    cuda_err = cudaStreamSynchronize(stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Copy result and deallocate device memory
    cuda_err = cudaMemcpy(&dst[0], dev_dst, sizeof(T)*m*n*k*batch,
            cudaMemcpyDeviceToHost);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaFree(dev_src1);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaFree(dev_src2);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaFree(dev_dst);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaStreamDestroy(stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
}
#endif // NNTILE_USE_CUDA

// Templated validation
template<typename T>
void validate(Index m, Index n, Index k, Index batch)
{
    using Y = typename T::repr_t;
    const Y eps = 2 * T::epsilon;
    // Init test input
    Scalar alpha = 1.0;
    Scalar beta = -1.0;
    std::vector<T> src1(k*batch), src2(m*n*k*batch), dst(m*n*k*batch);

    // Init random generator
    std::mt19937 rng(42);
    std::uniform_real_distribution<Y> dist(-1.0, 1.0);

    for(Index i = 0; i < k*batch; ++i)
    {
        src1[i] = dist(rng);
    }
    for(Index i = 0; i < m*n*k*batch; ++i)
    {
        src2[i] = dist(rng);
        dst[i] = dist(rng);
    }
    std::vector<T> dst_save(dst);
    std::cout << "Run kernel::add_fiber::cpu<" << T::short_name << ">\n";
    add_fiber::cpu<T>(m, n, k, batch, alpha, &src1[0], beta, &src2[0], &dst[0]);
    for(Index b = 0; b < batch; ++b)
    {
        for(Index i2 = 0; i2 < k; ++i2)
        {
            Y src1_val = Y(src1[i2+b*k]);
            for(Index i1 = 0; i1 < n; ++i1)
            {
                for(Index i0 = 0; i0 < m; ++i0)
                {
                    Index linear_idx = ((i1+b*n)*k+i2)*m+i0;
                    Y val_ref = alpha*src1_val + beta*Y(src2[linear_idx]);
                    Y val = Y(dst[linear_idx]);
                    if (std::abs(val_ref) > 10 * eps)
                    {
                        TEST_ASSERT(std::abs(val-val_ref)/std::abs(val_ref) <= eps);
                    }
                    else
                    {
                        TEST_ASSERT(std::abs(val-val_ref) <= eps);
                    }
                }
            }
        }
    }
    std::cout << "OK: kernel::add_fiber::cpu<" << T::short_name << ">\n";
#ifdef NNTILE_USE_CUDA
    // Check low-level CUDA kernel
    dst = dst_save;
    std::cout << "Run kernel::add_fiber::cuda<" << T::short_name << ">\n";
    run_cuda<T>(m, n, k, batch, alpha, src1, beta, src2, dst);
    for(Index b = 0; b < batch; ++b)
    {
        for(Index i2 = 0; i2 < k; ++i2)
        {
            Y src1_val = Y(src1[i2+b*k]);
            for(Index i1 = 0; i1 < n; ++i1)
            {
                for(Index i0 = 0; i0 < m; ++i0)
                {
                    Index linear_idx = ((i1+b*n)*k+i2)*m+i0;
                    Y val_ref = alpha*src1_val + beta*Y(src2[linear_idx]);
                    Y val = Y(dst[linear_idx]);
                    if (std::abs(val_ref) > 10 * eps)
                    {
                        TEST_ASSERT(std::abs(val-val_ref)/std::abs(val_ref) <= eps);
                    }
                    else
                    {
                        TEST_ASSERT(std::abs(val-val_ref) <= eps);
                    }
                }
            }
        }
    }
    std::cout << "OK: kernel::add_fiber::cuda<" << T::short_name << ">\n";
#endif // NNTILE_USE_CUDA
}

int main(int argc, char **argv)
{
    const Index test_m[] = {1, 5};
    const Index test_n[] = {1, 3};
    const Index test_k[] = {1, 10};
    const Index test_batch[] = {1, 4};

    for(Index m : test_m)
    {
        for(Index n : test_n)
        {
            for(Index k : test_k)
            {
                for(Index batch : test_batch)
                {
                    validate<fp32_t>(m, n, k, batch);
                    validate<fp64_t>(m, n, k, batch);
                    validate<bf16_t>(m, n, k, batch);
#ifdef NNTILE_USE_FP16
                    validate<fp16_t>(m, n, k, batch);
#endif // NNTILE_USE_FP16
                }
            }
        }
    }
}