/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/cpu/bias.cc
 * Bias operation on a buffer on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-03-26
 * */

#include "nntile/kernel/bias.hh"
#include "../testing.hh"
#include <vector>
#include <stdexcept>
#include <limits>
#include <iostream>

using namespace nntile;
using namespace nntile::kernel::bias;

#ifdef NNTILE_USE_CUDA
template<typename T>
void run_cuda(Index m, Index n, Index k, T alpha, const std::vector<T> &src,
        std::vector<T> &dst)
{
    // Copy to device
    T *dev_src, *dev_dst;
    cudaError_t cuda_err = cudaMalloc(&dev_src, sizeof(T)*m*n);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMalloc(&dev_dst, sizeof(T)*m*n*k);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_src, &src[0], sizeof(T)*m*n,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_dst, &dst[0], sizeof(T)*m*n*k,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Init stream
    cudaStream_t stream;
    cuda_err = cudaStreamCreate(&stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Launch low-level CUDA kernel
    cuda<T>(stream, m, n, k, alpha, dev_src, dev_dst);
    cuda_err = cudaStreamSynchronize(stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Copy result and deallocate device memory
    cuda_err = cudaMemcpy(&dst[0], dev_dst, sizeof(T)*m*n*k,
            cudaMemcpyDeviceToHost);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaFree(dev_src);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaFree(dev_dst);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaStreamDestroy(stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
}
#endif // NNTILE_USE_CUDA

// Templated validation
template<typename T>
void validate(Index m, Index n, Index k)
{
    constexpr T eps = std::numeric_limits<T>::epsilon();
    // Init test input
    std::vector<T> src(m*n), dst(m*n*k);
    for(Index i0 = 0; i0 < m; ++i0)
    {
        for(Index i1 = 0; i1 < n; ++i1)
        {
            for(Index i2 = 0; i2 < k; ++i2)
            {
                dst[(i1*k+i2)*m+i0] = T(i0+i1+i2) / T{10};
            }
        }
    }
    for(Index i0 = 0; i0 < m; ++i0)
    {
        for(Index i1 = 0; i1 < n; ++i1)
        {
            src[i1*m+i0] = T(i0+i1) / T{20};
        }
    }
    // Save original dst
    std::vector<T> dst_save(dst);
    // Check low-level CPU kernel
    std::cout << "Run kernel::bias::cpu<T>\n";
    cpu<T>(m, n, k, -2.0, &src[0], &dst[0]);
    for(Index i0 = 0; i0 < m; ++i0)
    {
        for(Index i1 = 0; i1 < n; ++i1)
        {
            // Check i2=0 at first, as it means val_ref = 0
            T val = dst[i1*k*m+i0];
            TEST_ASSERT(std::abs(val) <= 10*eps);
            for(Index i2 = 1; i2 < k; ++i2)
            {
                T val = dst[(i1*k+i2)*m+i0];
                T val_ref = T(i2) / T{10};
                TEST_ASSERT(std::abs(val/val_ref-T{1}) <= 10*eps);
            }
        }
    }
    std::cout << "OK: kernel::bias::cpu<T>\n";
#ifdef NNTILE_USE_CUDA
    // Check low-level CUDA kernel
    dst = dst_save;
    std::cout << "Run kernel::bias::cuda<T>\n";
    run_cuda<T>(m, n, k, -2.0, src, dst);
    for(Index i0 = 0; i0 < m; ++i0)
    {
        for(Index i1 = 0; i1 < n; ++i1)
        {
            // Check i2=0 at first, as it means val_ref = 0
            T val = dst[i1*k*m+i0];
            TEST_ASSERT(std::abs(val) <= 10*eps);
            for(Index i2 = 1; i2 < k; ++i2)
            {
                T val = dst[(i1*k+i2)*m+i0];
                T val_ref = T(i2) / T{10};
                if(std::abs(val/val_ref-T{1}) <= 10*eps);
            }
        }
    }
    std::cout << "OK: kernel::bias::cuda<T>\n";
#endif // NNTILE_USE_CUDA
}

template<typename T>
void validate(T val, Index num_elements)
{
    constexpr T eps = std::numeric_limits<T>::epsilon();
    // Init test input
    std::vector<T> x(num_elements);
    for (Index i = 0; i < num_elements; ++i)
        x[i] = i / rand();
    std::vector<T> y(x);
    // Check low-level kernel
    std::cout << "Run kernel::bias::cpu<T>\n";
    cpu<T>(val, num_elements, &x[0]);
    for (Index i = 0; i < num_elements; ++i)
        TEST_ASSERT(y[i] + val == x[i]);
    std::cout << "OK: kernel::bias::cpu<T>\n";
}

int main(int argc, char **argv)
{
    // Validate bias for middle axis
    validate<fp32_t>(1, 9, 10);
    validate<fp32_t>(8, 9, 1);
    validate<fp32_t>(8, 1, 10);
    validate<fp32_t>(4, 7, 8);

    validate<fp64_t>(1, 9, 10);
    validate<fp64_t>(8, 9, 1);
    validate<fp64_t>(8, 1, 10);
    validate<fp64_t>(4, 7, 8);

    // Validate bias of all tensor elements
    validate<fp32_t>(10, 100);
    validate<fp32_t>(-1, 10);
    validate<fp64_t>(10.5, 1000);
    validate<fp64_t>(-1, 10);

    return 0;
}

