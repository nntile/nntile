/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/sum_slice.cc
 * Sums over fibers into a slice of a buffer
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/sum_slice.hh"
#include "../testing.hh"
#include <vector>
#include <stdexcept>
#include <limits>
#include <cmath>
#include <iostream>

using namespace nntile;
using namespace nntile::kernel::sum_slice;

#ifdef NNTILE_USE_CUDA
template<typename T>
void run_cuda(Index m, Index n, Index k, Scalar alpha, const std::vector<T> &src,
        Scalar beta, std::vector<T> &sum_dst)
{
    // Copy to device
    T *dev_src, *dev_sum_dst;
    cudaError_t cuda_err = cudaMalloc(&dev_src, sizeof(T)*m*n*k);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMalloc(&dev_sum_dst, sizeof(T)*m*n);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_src, &src[0], sizeof(T)*m*n*k,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_sum_dst, &sum_dst[0], sizeof(T)*m*n,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Init stream
    cudaStream_t stream;
    cuda_err = cudaStreamCreate(&stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Launch low-level kernel
    cuda<T>(stream, m, n, k, alpha, dev_src, beta, dev_sum_dst);
    cuda_err = cudaStreamSynchronize(stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Copy result and deallocate device memory
    cuda_err = cudaMemcpy(&sum_dst[0], dev_sum_dst, sizeof(T)*m*n,
            cudaMemcpyDeviceToHost);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaFree(dev_src);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaFree(dev_sum_dst);
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
    const Y eps = T::epsilon();
    // Init test input
    std::vector<T> src(m*n*k), sum_dst(m*n);
    for(Index i0 = 0; i0 < m; ++i0)
    {
        for(Index i1 = 0; i1 < n; ++i1)
        {
            for(Index i2 = 0; i2 < k; ++i2)
            {
                src[(i1*k+i2)*m+i0] = Y(i0+i1+i2) / Y{10};
            }
            sum_dst[i1*m+i0] = Y{1.0};
        }
    }
    std::vector<T> sum_copy(sum_dst);
    // Check low-level kernel
    std::cout << "Run kernel::sum_slice::cpu<" << T::type_repr << ">\n";
    cpu<T>(m, n, k, alpha, &src[0], beta, &sum_dst[0]);
    for(Index i0 = 0; i0 < m; ++i0)
    {
        for(Index i1 = 0; i1 < n; ++i1)
        {
            Index a = i0 + i1;
            Y sum_ref = k * (2*a+k-1) / 2 / Y{10};
            sum_ref = alpha*sum_ref + beta;
            Y sum(sum_dst[i1*m+i0]);
            if(sum_ref == Y{0})
            {
                TEST_ASSERT(std::abs(sum) <= 10*eps);
            }
            else
            {
                TEST_ASSERT(std::abs(sum/sum_ref-Y{1}) <= 10*eps);
            }
        }
    }
    std::cout << "OK: kernel::sum_slice::cpu<" << T::type_repr << ">\n";
#ifdef NNTILE_USE_CUDA
    // Check low-level CUDA kernel
    std::vector<T> sum_cuda(sum_copy);
    std::cout << "Run kernel::sum_slice::cuda<" << T::type_repr << ">\n";
    run_cuda<T>(m, n, k, alpha, src, beta, sum_cuda);
    for(Index i0 = 0; i0 < m; ++i0)
    {
        for(Index i1 = 0; i1 < n; ++i1)
        {
            Index i = (i1*m+i0);
            if(Y(sum_dst[i]) == Y{0})
            {
                TEST_ASSERT(Y(sum_cuda[i]) == Y{0});
            }
            else
            {
                TEST_ASSERT(std::abs(Y(sum_cuda[i])/Y(sum_dst[i])-Y{1}) <= 10*eps);
            }

        }
    }
    std::cout << "OK: kernel::sum_slice::cuda<" << T::type_repr << ">\n";
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
