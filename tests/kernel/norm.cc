/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/norm.cc
 * Norm of a buffer on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-18
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
void run_cuda(Index m, Index n, Index k, T alpha, const std::vector<T> &src,
        T beta, std::vector<T> &norm_dst)
{
    // Copy to device
    T *dev_src, *dev_norm_dst;
    cudaError_t cuda_err = cudaMalloc(&dev_src, sizeof(T)*m*n*k);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMalloc(&dev_norm_dst, sizeof(T)*m*n);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_src, &src[0], sizeof(T)*m*n*k,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_norm_dst, &norm_dst[0], sizeof(T)*m*n,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Init stream
    cudaStream_t stream;
    cuda_err = cudaStreamCreate(&stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Launch low-level kernel
    cuda<T>(stream, m, n, k, alpha, dev_src, beta, dev_norm_dst);
    cuda_err = cudaStreamSynchronize(stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Copy result and deallocate device memory
    cuda_err = cudaMemcpy(&norm_dst[0], dev_norm_dst, sizeof(T)*m*n,
            cudaMemcpyDeviceToHost);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaFree(dev_src);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaFree(dev_norm_dst);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaStreamDestroy(stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
}
#endif // NNTILE_USE_CUDA

// Templated validation
template<typename T>
void validate(Index m, Index n, Index k, T alpha, T beta)
{
    constexpr T eps = std::numeric_limits<T>::epsilon();
    // Init test input
    std::vector<T> src(m*n*k), norm_dst(m*n);
    for(Index i0 = 0; i0 < m; ++i0)
    {
        for(Index i1 = 0; i1 < n; ++i1)
        {
            for(Index i2 = 0; i2 < k; ++i2)
            {
                src[(i1*k+i2)*m+i0] = std::sqrt(T(i0+i1+i2) / T{10});
            }
            norm_dst[i1*m+i0] = T{1.0};
        }
    } 
    std::vector<T> norm_copy(norm_dst);
    // Check low-level kernel
    std::cout << "Run kernel::norm::cpu<T>\n";
    cpu<T>(m, n, k, alpha, &src[0], beta, &norm_dst[0]);
    for(Index i0 = 0; i0 < m; ++i0)
    {
        for(Index i1 = 0; i1 < n; ++i1)
        {
            Index a = i0 + i1;
            T norm_ref = k * (2*a+k-1) / 2 / T{10};
            norm_ref = std::sqrt(norm_ref);
            norm_ref = std::hypot(alpha*norm_ref, beta);
            T norm = norm_dst[i1*m+i0];
            if(norm_ref == T{0})
            {
                TEST_ASSERT(std::abs(norm) <= 10*eps);
            }
            else
            {
                TEST_ASSERT(std::abs(norm/norm_ref-T{1}) <= 10*eps);
            }
            
        }
    }
    std::cout << "OK: kernel::norm::cpu<T>\n";
#ifdef NNTILE_USE_CUDA
    // Check low-level CUDA kernel
    std::vector<T> norm_cuda(norm_copy);
    std::cout << "Run kernel::norm::cuda<T>\n";
    run_cuda<T>(m, n, k, alpha, src, beta, norm_cuda);
    for(Index i0 = 0; i0 < m; ++i0)
    {
        for(Index i1 = 0; i1 < n; ++i1)
        {
            Index i = (i1*m+i0);
            if(norm_dst[i] == T{0})
            {
                TEST_ASSERT(norm_cuda[i] == T{0});   
            }
            else
            {   
                TEST_ASSERT(std::abs(norm_cuda[i]/norm_dst[i]-T{1})
                        <= 10*eps); 
            }
            
        }
    }
    std::cout << "OK: kernel::norm::cuda<T>\n";
#endif // NNTILE_USE_CUDA
}

int main(int argc, char **argv)
{
    validate<fp32_t>(1, 9, 10, 1.0, 1.0);
    validate<fp32_t>(8, 9, 1, 1.0, -1.0);
    validate<fp32_t>(8, 1, 10, -1.0, 1.0);
    validate<fp32_t>(4, 7, 8, 0.0, 2.0);
    validate<fp64_t>(1, 9, 10, 2.0, 0.0);
    validate<fp64_t>(8, 9, 1, 1.0, 1.0);
    validate<fp64_t>(8, 1, 10, -1.0, -1.0);
    validate<fp64_t>(4, 7, 8, 2.5, 1.25);
    return 0;
}

