/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/maxsumexp.cc
 * Max and sums of exponents of a buffer on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-12-07
 * */

#include "nntile/kernel/maxsumexp.hh"
#include "../testing.hh"
#include <vector>
#include <stdexcept>
#include <limits>
#include <cmath>
#include <iostream>

using namespace nntile;
using namespace nntile::kernel::maxsumexp;

#ifdef NNTILE_USE_CUDA
template<typename T>
void run_cuda(Index m, Index n, Index k, const std::vector<T> &src,
        std::vector<T> &maxsumexp)
{
    // Copy to device
    T *dev_src, *dev_maxsumexp;
    cudaError_t cuda_err = cudaMalloc(&dev_src, sizeof(T)*m*n*k);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMalloc(&dev_maxsumexp, sizeof(T)*2*m*n);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_src, &src[0], sizeof(T)*m*n*k,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_maxsumexp, &maxsumexp[0], sizeof(T)*2*m*n,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Init stream
    cudaStream_t stream;
    cuda_err = cudaStreamCreate(&stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Launch low-level kernel
    cuda<T>(stream, m, n, k, dev_src, dev_maxsumexp);
    cuda_err = cudaStreamSynchronize(stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Copy result and deallocate device memory
    cuda_err = cudaMemcpy(&maxsumexp[0], dev_maxsumexp, sizeof(T)*2*m*n,
            cudaMemcpyDeviceToHost);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaFree(dev_src);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaFree(dev_maxsumexp);
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
    std::vector<T> src(m*n*k), maxsumexp(2*m*n);
    for(Index i0 = 0; i0 < m; ++i0)
    {
        for(Index i1 = 0; i1 < n; ++i1)
        {
            for(Index i2 = 0; i2 < k; ++i2)
            {
                src[(i1*k+i2)*m+i0] = T(i0+i1+i2) / T{10};
            }
        }
    }
    std::vector<T> maxsumexp_copy(maxsumexp);
    // Check low-level kernel
    std::cout << "Run kernel::maxsumexp::cpu<T>\n";
    cpu<T>(m, n, k, &src[0], &maxsumexp[0]);
    for(Index i0 = 0; i0 < m; ++i0)
    {
        for(Index i1 = 0; i1 < n; ++i1)
        {
            Index a = i0 + i1;
            T max_ref = T(a + k - 1) / T{10};
            T max = maxsumexp[2*(i1*m+i0)];
            TEST_ASSERT(max == max_ref);
            T sum_ref = 0;
            for(Index i2 = 0; i2 < k; ++i2)
            {
                sum_ref += std::exp(T(i2-k+1)/T{10});
            }
            T sum = maxsumexp[2*(i1*m+i0)+1];
            TEST_ASSERT(std::abs(sum/sum_ref-T{1}) <= 10*eps);
        }
    }
    std::cout << "OK: kernel::maxsumexp::cpu<T>\n";
#ifdef NNTILE_USE_CUDA
    // Check low-level CUDA kernel
    maxsumexp = maxsumexp_copy;
    std::cout << "Run kernel::maxsumexp::cuda<T>\n";
    run_cuda<T>(m, n, k, src, maxsumexp);
    for(Index i0 = 0; i0 < m; ++i0)
    {
        for(Index i1 = 0; i1 < n; ++i1)
        {
            Index a = i0 + i1;
            T max_ref = T(a + k - 1) / T{10};
            T max = maxsumexp[2*(i1*m+i0)];
            TEST_ASSERT(max == max_ref);
            T sum_ref = 0;
            for(Index i2 = 0; i2 < k; ++i2)
            {
                sum_ref += std::exp(T(i2-k+1)/T{10});
            }
            T sum = maxsumexp[2*(i1*m+i0)+1];
            TEST_ASSERT(std::abs(sum/sum_ref-T{1}) <= 10*eps);
        }
    }
    std::cout << "OK: kernel::maxsumexp::cuda<T>\n";
#endif // NNTILE_USE_CUDA
    // Check low-level kernel even more
    maxsumexp_copy = maxsumexp;
    std::cout << "Run kernel::maxsumexp::cpu<T>\n";
    cpu<T>(m, n, k, &src[0], &maxsumexp[0]);
    for(Index i0 = 0; i0 < m; ++i0)
    {
        for(Index i1 = 0; i1 < n; ++i1)
        {
            Index i = 2 * (i1*m+i0);
            TEST_ASSERT(maxsumexp[i] == maxsumexp_copy[i]);
            TEST_ASSERT(std::abs(maxsumexp[i+1]/maxsumexp_copy[i+1]
                        -T{2}) <= 10*eps);
        }
    }
    std::cout << "OK: kernel::maxsumexp::cpu<T>\n";
#ifdef NNTILE_USE_CUDA
    // Check low-level CUDA kernel
    maxsumexp = maxsumexp_copy;
    std::cout << "Run kernel::maxsumexp::cuda<T>\n";
    run_cuda<T>(m, n, k, src, maxsumexp);
    for(Index i0 = 0; i0 < m; ++i0)
    {
        for(Index i1 = 0; i1 < n; ++i1)
        {
            Index i = 2 * (i1*m+i0);
            TEST_ASSERT(maxsumexp[i] == maxsumexp_copy[i]);
            TEST_ASSERT(std::abs(maxsumexp[i+1]/maxsumexp_copy[i+1]
                        -T{2}) <= 10*eps);
        }
    }
    std::cout << "OK: kernel::maxsumexp::cuda<T>\n";
#endif // NNTILE_USE_CUDA
}

int main(int argc, char **argv)
{
    validate<fp32_t>(1, 9, 10);
    validate<fp32_t>(8, 9, 1);
    validate<fp32_t>(8, 1, 10);
    validate<fp32_t>(4, 7, 8);
    validate<fp64_t>(1, 9, 10);
    validate<fp64_t>(8, 9, 1);
    validate<fp64_t>(8, 1, 10);
    validate<fp64_t>(4, 7, 8);
    return 0;
}

