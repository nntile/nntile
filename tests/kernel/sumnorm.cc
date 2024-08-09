/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/sumnorm.cc
 * Sum and Euclidean norm of a buffer on CPU
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/sumnorm.hh"
#include "../testing.hh"
#include <vector>
#include <stdexcept>
#include <limits>
#include <cmath>
#include <iostream>

using namespace nntile;
using namespace nntile::kernel::sumnorm;

#ifdef NNTILE_USE_CUDA
template<typename T>
void run_cuda(Index m, Index n, Index k, const std::vector<T> &src,
        std::vector<T> &sumnorm)
{
    // Copy to device
    T *dev_src, *dev_sumnorm;
    cudaError_t cuda_err = cudaMalloc(&dev_src, sizeof(T)*m*n*k);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMalloc(&dev_sumnorm, sizeof(T)*2*m*n);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_src, &src[0], sizeof(T)*m*n*k,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_sumnorm, &sumnorm[0], sizeof(T)*2*m*n,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Init stream
    cudaStream_t stream;
    cuda_err = cudaStreamCreate(&stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Launch low-level kernel
    cuda<T>(stream, m, n, k, dev_src, dev_sumnorm);
    cuda_err = cudaStreamSynchronize(stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Copy result and deallocate device memory
    cuda_err = cudaMemcpy(&sumnorm[0], dev_sumnorm, sizeof(T)*2*m*n,
            cudaMemcpyDeviceToHost);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaFree(dev_src);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaFree(dev_sumnorm);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaStreamDestroy(stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
}
#endif // NNTILE_USE_CUDA

// Templated validation
template<typename T>
void validate(Index m, Index n, Index k)
{
    using Y = typename T::repr_t;
    const Y eps = T::epsilon();
    // Init test input
    std::vector<T> src(m*n*k), sumnorm(2*m*n);
    for(Index i0 = 0; i0 < m; ++i0)
    {
        for(Index i1 = 0; i1 < n; ++i1)
        {
            for(Index i2 = 0; i2 < k; ++i2)
            {
                src[(i1*k+i2)*m+i0] = Y(i0+i1+i2) / Y{10};
            }
        }
    }
    std::vector<T> sumnorm_copy(sumnorm);
    // Check low-level kernel
    std::cout << "Run kernel::sumnorm::cpu<" << T::type_repr << ">\n";
    cpu<T>(m, n, k, &src[0], &sumnorm[0]);
    for(Index i0 = 0; i0 < m; ++i0)
    {
        for(Index i1 = 0; i1 < n; ++i1)
        {
            Index a = i0 + i1;
            Y sum_ref = k * (2*a+k-1) / 2 / Y{10};
            Y sum(sumnorm[2*(i1*m+i0)]);
            if(sum_ref == Y{0})
            {
                TEST_ASSERT(std::abs(sum) <= 10*eps);
            }
            else
            {
                TEST_ASSERT(std::abs(sum/sum_ref-Y{1}) <= 10*eps);
            }
            Y norm_sqr_ref = k * (2*(a-1)*a+(2*a+k-1)*(2*a+2*k-1)) / 6
                / Y{100};
            Y norm(sumnorm[2*(i1*m+i0)+1]);
            Y norm_sqr = norm * norm;
            if(norm_sqr_ref == Y{0})
            {
                TEST_ASSERT(norm_sqr <= 10*eps);
            }
            else
            {
                TEST_ASSERT(std::abs(norm_sqr/norm_sqr_ref-Y{1}) <= 10*eps);
            }
        }
    }
    std::cout << "OK: kernel::sumnorm::cpu<" << T::type_repr << ">\n";
#ifdef NNTILE_USE_CUDA
    // Check low-level CUDA kernel
    sumnorm = sumnorm_copy;
    std::cout << "Run kernel::sumnorm::cuda<" << T::type_repr << ">\n";
    run_cuda<T>(m, n, k, src, sumnorm);
    for(Index i0 = 0; i0 < m; ++i0)
    {
        for(Index i1 = 0; i1 < n; ++i1)
        {
            Index a = i0 + i1;
            Y sum_ref = k * (2*a+k-1) / 2 / Y{10};
            Y sum(sumnorm[2*(i1*m+i0)]);
            if(sum_ref == Y{0})
            {
                TEST_ASSERT(std::abs(sum) <= 10*eps);
            }
            else
            {
                TEST_ASSERT(std::abs(sum/sum_ref-Y{1}) <= 10*eps);
            }
            Y norm_sqr_ref = k * (2*(a-1)*a+(2*a+k-1)*(2*a+2*k-1)) / 6
                / Y{100};
            Y norm(sumnorm[2*(i1*m+i0)+1]);
            Y norm_sqr = norm * norm;
            if(norm_sqr_ref == Y{0})
            {
                TEST_ASSERT(norm_sqr <= 10*eps);
            }
            else
            {
                TEST_ASSERT(std::abs(norm_sqr/norm_sqr_ref-Y{1}) <= 10*eps);
            }
        }
    }
    std::cout << "OK: kernel::sumnorm::cuda<" << T::type_repr << ">\n";
#endif // NNTILE_USE_CUDA
    // Check low-level kernel even more
    sumnorm_copy = sumnorm;
    std::cout << "Run kernel::sumnorm::cpu<" << T::type_repr << ">\n";
    cpu<T>(m, n, k, &src[0], &sumnorm[0]);
    for(Index i0 = 0; i0 < m; ++i0)
    {
        for(Index i1 = 0; i1 < n; ++i1)
        {
            Index i = 2 * (i1*m+i0);
            if(Y(sumnorm_copy[i]) == Y{0})
            {
                TEST_ASSERT(Y(sumnorm[i]) == Y{0});
            }
            else
            {
                TEST_ASSERT(std::abs(Y(sumnorm[i])/Y(sumnorm_copy[i])-Y{2})
                        <= 10*eps);
            }
            if(Y(sumnorm_copy[i+1]) == Y{0})
            {
                TEST_ASSERT(Y(sumnorm[i+1]) == Y{0});
            }
            else
            {
                TEST_ASSERT(std::abs(Y(sumnorm[i+1])/Y(sumnorm_copy[i+1])
                        -std::sqrt(Y{2})) <= 10*eps);
            }
        }
    }
    std::cout << "OK: kernel::sumnorm::cpu<" << T::type_repr << ">\n";
#ifdef NNTILE_USE_CUDA
    // Check low-level CUDA kernel
    sumnorm = sumnorm_copy;
    std::cout << "Run kernel::sumnorm::cuda<" << T::type_repr << ">\n";
    run_cuda<T>(m, n, k, src, sumnorm);
    for(Index i0 = 0; i0 < m; ++i0)
    {
        for(Index i1 = 0; i1 < n; ++i1)
        {
            Index i = 2 * (i1*m+i0);
            if(Y(sumnorm_copy[i]) == Y{0})
            {
                TEST_ASSERT(Y(sumnorm[i]) == Y{0});
            }
            else
            {
                TEST_ASSERT(std::abs(Y(sumnorm[i])/Y(sumnorm_copy[i])-Y{2})
                        <= 10*eps);
            }
            if(Y(sumnorm_copy[i+1]) == Y{0})
            {
                TEST_ASSERT(Y(sumnorm[i+1]) == Y{0});
            }
            else
            {
                TEST_ASSERT(std::abs(Y(sumnorm[i+1])/Y(sumnorm_copy[i+1])
                        -std::sqrt(Y{2})) <= 10*eps);
            }
        }
    }
    std::cout << "OK: kernel::sumnorm::cuda<" << T::type_repr << ">\n";
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
