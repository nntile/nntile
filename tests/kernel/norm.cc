/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/norm.cc
 * Euclidian norm of a buffer on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-11-28
 * */

#include "nntile/kernel/norm.hh"
#include "../testing.hh"
#include <vector>
#include <stdexcept>
#include <limits>
#include <cmath>
#include <iostream>

using namespace nntile;
using namespace nntile::kernel::norm;

#ifdef NNTILE_USE_CUDA
template<typename T>
void run_cuda(Index m, Index n, Index k, const std::vector<T> &src,
        std::vector<T> &norm)
{
    // Copy to device
    T *dev_src, *dev_norm;
    cudaError_t cuda_err = cudaMalloc(&dev_src, sizeof(T)*m*n*k);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMalloc(&dev_norm, sizeof(T)*m*n);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_src, &src[0], sizeof(T)*m*n*k,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_norm, &norm[0], sizeof(T)*m*n,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Init stream
    cudaStream_t stream;
    cuda_err = cudaStreamCreate(&stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Launch low-level kernel
    cuda<T>(stream, m, n, k, dev_src, dev_norm);
    cuda_err = cudaStreamSynchronize(stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Copy result and deallocate device memory
    cuda_err = cudaMemcpy(&norm[0], dev_norm, sizeof(T)*m*n,
            cudaMemcpyDeviceToHost);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaFree(dev_src);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaFree(dev_norm);
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
    std::vector<T> src(m*n*k), norm(m*n);
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
    std::vector<T> norm_copy(norm);
    // Check low-level kernel
    std::cout << "Run kernel::norm::cpu<T>\n";
    cpu<T>(m, n, k, &src[0], &norm[0]);
    for(Index i0 = 0; i0 < m; ++i0)
    {
        for(Index i1 = 0; i1 < n; ++i1)
        {
            Index a = i0 + i1;
            T norm_sqr_ref = k * (2*(a-1)*a+(2*a+k-1)*(2*a+2*k-1)) / 6
                / T{100};
            T norm_val = norm[i1*m+i0];
            T norm_sqr = norm_val * norm_val;
            if(norm_sqr_ref == T{0})
            {
                TEST_ASSERT(norm_sqr <= 10*eps);
            }
            else
            {
                TEST_ASSERT(std::abs(norm_sqr/norm_sqr_ref-T{1}) <= 10*eps);
            }
        }
    }
    std::cout << "OK: kernel::norm::cpu<T>\n";
#ifdef NNTILE_USE_CUDA
    // Check low-level CUDA kernel
    norm = norm_copy;
    std::cout << "Run kernel::norm::cuda<T>\n";
    run_cuda<T>(m, n, k, src, norm);
    for(Index i0 = 0; i0 < m; ++i0)
    {
        for(Index i1 = 0; i1 < n; ++i1)
        {
            Index a = i0 + i1;
            T norm_sqr_ref = k * (2*(a-1)*a+(2*a+k-1)*(2*a+2*k-1)) / 6
                / T{100};
            T norm_val = norm[i1*m+i0];
            T norm_sqr = norm_val * norm_val;
            if(norm_sqr_ref == T{0})
            {
                TEST_ASSERT(norm_sqr <= 10*eps);
            }
            else
            {
                TEST_ASSERT(std::abs(norm_sqr/norm_sqr_ref-T{1}) <= 10*eps);
            }
        }
    }
    std::cout << "OK: kernel::norm::cuda<T>\n";
#endif // NNTILE_USE_CUDA
    // Check low-level kernel even more
    norm_copy = norm;
    std::cout << "Run kernel::norm::cpu<T>\n";
    cpu<T>(m, n, k, &src[0], &norm[0]);
    for(Index i0 = 0; i0 < m; ++i0)
    {
        for(Index i1 = 0; i1 < n; ++i1)
        {
            Index i = i1*m + i0;
            if(norm_copy[i] == T{0})
            {
                TEST_ASSERT(norm[i] == T{0});
            }
            else
            {
                TEST_ASSERT(std::abs(norm[i]/norm_copy[i]
                        -std::sqrt(T{2})) <= 10*eps);
            }
        }
    }
    std::cout << "OK: kernel::norm::cpu<T>\n";
#ifdef NNTILE_USE_CUDA
    // Check low-level CUDA kernel
    norm = norm_copy;
    std::cout << "Run kernel::norm::cuda<T>\n";
    run_cuda<T>(m, n, k, src, norm);
    for(Index i0 = 0; i0 < m; ++i0)
    {
        for(Index i1 = 0; i1 < n; ++i1)
        {
            Index i = i1*m + i0;
            if(norm_copy[i] == T{0})
            {
                TEST_ASSERT(norm[i] == T{0});
            }
            else
            {
                TEST_ASSERT(std::abs(norm[i]/norm_copy[i]
                        -std::sqrt(T{2})) <= 10*eps);
            }
        }
    }
    std::cout << "OK: kernel::norm::cuda<T>\n";
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

