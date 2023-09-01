/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/softmax_inplace.cc
 * softmax_inplace operation for a buffer on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-07-02
 * */

#include "nntile/kernel/softmax_inplace.hh"
#include "../testing.hh"
#include <vector>
#include <stdexcept>
#include <limits>
#include <cmath>
#include <iostream>

using namespace nntile;
using namespace nntile::kernel::softmax_inplace;

#ifdef NNTILE_USE_CUDA
template<typename T>
void run_cuda(Index m, Index n, Index k, const std::vector<T> &maxsumexp,
        std::vector<T> &dst)
{
    // Copy to device
    T *dev_maxsumexp, *dev_dst;
    cudaError_t cuda_err = cudaMalloc(&dev_maxsumexp, sizeof(T)*2*m*n);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMalloc(&dev_dst, sizeof(T)*m*n*k);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_maxsumexp, &maxsumexp[0], sizeof(T)*2*m*n,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_dst, &dst[0], sizeof(T)*m*n*k,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Init stream
    cudaStream_t stream;
    cuda_err = cudaStreamCreate(&stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Launch low-level kernel
    cuda<T>(stream, m, n, k, dev_maxsumexp, dev_dst);
    // Wait for result and destroy stream
    cuda_err = cudaStreamSynchronize(stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaStreamDestroy(stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Copy result and deallocate device memory
    cuda_err = cudaMemcpy(&dst[0], dev_dst, sizeof(T)*m*n*k,
            cudaMemcpyDeviceToHost);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaFree(dev_maxsumexp);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaFree(dev_dst);
    TEST_ASSERT(cuda_err == cudaSuccess);
}
#endif // NNTILE_USE_CUDA

// Templated validation
template<typename T>
void validate(Index m, Index n, Index k)
{
    constexpr T epsilon = std::numeric_limits<T>::epsilon();
    // Init test input
    std::vector<T> maxsumexp(2*m*n), dst(m*n*k);
    for(Index i0 = 0; i0 < m; ++i0)
    {
        for(Index i1 = 0; i1 < n; ++i1)
        {
            for(Index i2 = 0; i2 < k; ++i2)
            {
                dst[(i1*k+i2)*m+i0] = T(i0+i1+i2) / T{100};
            }
        }
    }
    for(Index i0 = 0; i0 < m; ++i0)
    {
        for(Index i1 = 0; i1 < n; ++i1)
        {
            T max = T(i0+i1+k-1) / T{100};
            maxsumexp[2*(i1*m+i0)] = max;
            maxsumexp[2*(i1*m+i0)+1] = T(i0+i1+1) / T{100};
        }
    }
    std::vector<T> dst_save(dst);
    // Check low-level kernel
    std::cout << "Run kernel::softmax_inplace::cpu<T>\n";
    cpu<T>(m, n, k, &maxsumexp[0], &dst[0]);
    for(Index i0 = 0; i0 < m; ++i0)
    {
        for(Index i1 = 0; i1 < n; ++i1)
        {
            for(Index i2 = 0; i2 < k; ++i2)
            {
                T val = dst[(i1*k+i2)*m+i0];
                T val_ref = std::exp(T(i2-k+1) / T{100}) * T{100} / T(i0+i1+1);
                T tmp = std::abs(val - val_ref);
                TEST_ASSERT(tmp/val_ref < 100*epsilon);
            }
        }
    }
    std::cout << "OK: kernel::softmax_inplace::cpu<T>\n";
#ifdef NNTILE_USE_CUDA
    // Check low-level CUDA kernel
    dst = dst_save;
    std::cout << "Run kernel::softmax_inplace::cuda<T>\n";
    run_cuda<T>(m, n, k, maxsumexp, dst);
    for(Index i0 = 0; i0 < m; ++i0)
    {
        for(Index i1 = 0; i1 < n; ++i1)
        {
            for(Index i2 = 0; i2 < k; ++i2)
            {
                T val = dst[(i1*k+i2)*m+i0];
                T val_ref = std::exp(T(i2-k+1) / T{100}) * T{100} / T(i0+i1+1);
                T tmp = std::abs(val - val_ref);
                TEST_ASSERT(tmp/val_ref < 100*epsilon);
            }
        }
    }
    std::cout << "OK: kernel::softmax_inplace::cuda<T>\n";
#endif // NNTILE_USE_CUDA
}

int main(int argc, char **argv)
{
    validate<fp32_t>(1, 9, 11);
    validate<fp32_t>(8, 1, 11);
    validate<fp32_t>(8, 9, 1);
    validate<fp32_t>(1, 450, 450);
    validate<fp32_t>(450, 1, 450);
    validate<fp32_t>(450, 450, 1);
    validate<fp64_t>(1, 9, 11);
    validate<fp64_t>(8, 1, 11);
    validate<fp64_t>(8, 9, 1);
    validate<fp64_t>(1, 450, 450);
    validate<fp64_t>(450, 1, 450);
    validate<fp64_t>(450, 450, 1);
    return 0;
}

