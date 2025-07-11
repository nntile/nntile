/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/norm_slice.cc
 * Euclidean norms of fibers into a slice of a buffer
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/norm_slice.hh"
#include "../testing.hh"
#include <vector>
#include <stdexcept>
#include <limits>
#include <cmath>
#include <iostream>

using namespace nntile;
using namespace nntile::kernel::norm_slice;

#ifdef NNTILE_USE_CUDA
template<typename T>
void run_cuda(Index m, Index n, Index k, Scalar alpha,
        const std::vector<T> &src, Scalar beta, std::vector<T> &dst)
{
    // Copy to device
    T *dev_src, *dev_dst;
    cudaError_t cuda_err = cudaMalloc(&dev_src, sizeof(T)*m*n*k);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMalloc(&dev_dst, sizeof(T)*m*n);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_src, &src[0], sizeof(T)*m*n*k,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_dst, &dst[0], sizeof(T)*m*n,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Init stream
    cudaStream_t stream;
    cuda_err = cudaStreamCreate(&stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Launch low-level kernel
    cuda<T>(stream, m, n, k, alpha, dev_src, beta, dev_dst);
    cuda_err = cudaStreamSynchronize(stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Copy result and deallocate device memory
    cuda_err = cudaMemcpy(&dst[0], dev_dst, sizeof(T)*m*n,
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
void validate(Index m, Index n, Index k, Scalar alpha, Scalar beta)
{
    using Y = typename T::repr_t;
    const Y eps = T::epsilon;
    // Init test input
    std::vector<T> src(m*n*k), dst(m*n);
    for(Index i0 = 0; i0 < m; ++i0)
    {
        for(Index i1 = 0; i1 < n; ++i1)
        {
            for(Index i2 = 0; i2 < k; ++i2)
            {
                src[(i1*k+i2)*m+i0] = std::sqrt(Y(i0+i1+i2) / Y{10});
            }
            dst[i1*m+i0] = Y{1.0};
        }
    }
    std::vector<T> dst_copy(dst);
    // Check low-level kernel
    std::cout << "Run kernel::norm_slice::cpu<" << T::short_name << ">\n";
    cpu<T>(m, n, k, alpha, &src[0], beta, &dst[0]);
    for(Index i0 = 0; i0 < m; ++i0)
    {
        for(Index i1 = 0; i1 < n; ++i1)
        {
            Index a = i0 + i1;
            Y ref = k * (2*a+k-1) / 2 / Y{10};
            ref = std::sqrt(ref);
            ref = std::hypot(alpha*ref, beta);
            Y val{dst[i1*m+i0]};
            if(ref == Y{0})
            {
                TEST_ASSERT(std::abs(val) <= 10*eps);
            }
            else
            {
                TEST_ASSERT(std::abs(val/ref-Y{1}) <= 10*eps);
            }

        }
    }
    std::cout << "OK: kernel::norm_slice::cpu<" << T::short_name << ">\n";
#ifdef NNTILE_USE_CUDA
    // Check low-level CUDA kernel
    std::vector<T> dst_cuda(dst_copy);
    std::cout << "Run kernel::norm_slice::cuda<" << T::short_name << ">\n";
    run_cuda<T>(m, n, k, alpha, src, beta, dst_cuda);
    for(Index i0 = 0; i0 < m; ++i0)
    {
        for(Index i1 = 0; i1 < n; ++i1)
        {
            Index i = (i1*m+i0);
            if(Y(dst[i]) == Y{0})
            {
                TEST_ASSERT(Y(dst_cuda[i]) == Y{0});
            }
            else
            {
                TEST_ASSERT(std::abs(Y(dst_cuda[i])/Y(dst[i])-Y{1})
                        <= 10*eps);
            }

        }
    }
    std::cout << "OK: kernel::norm_slice::cuda<" << T::short_name << ">\n";
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
