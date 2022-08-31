/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/sumnorm.cc
 * Sum and Euclidian norm of a buffer on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-31
 * */

#include "nntile/kernel/sumnorm/cpu.hh"
#include "nntile/defs.h"
#ifdef NNTILE_USE_CUDA
#   include "nntile/kernel/sumnorm/cuda.hh"
#endif // NNTILE_USE_CUDA
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
    if(cuda_err != cudaSuccess)
    {
        throw std::runtime_error("CUDA error");
    }
    cuda_err = cudaMalloc(&dev_sumnorm, sizeof(T)*2*m*n);
    if(cuda_err != cudaSuccess)
    {
        throw std::runtime_error("CUDA error");
    }
    cuda_err = cudaMemcpy(dev_src, &src[0], sizeof(T)*m*n*k,
            cudaMemcpyHostToDevice);
    if(cuda_err != cudaSuccess)
    {
        throw std::runtime_error("CUDA error");
    }
    cuda_err = cudaMemcpy(dev_sumnorm, &sumnorm[0], sizeof(T)*2*m*n,
            cudaMemcpyHostToDevice);
    if(cuda_err != cudaSuccess)
    {
        throw std::runtime_error("CUDA error");
    }
    // Init stream
    cudaStream_t stream;
    cuda_err = cudaStreamCreate(&stream);
    if(cuda_err != cudaSuccess)
    {
        throw std::runtime_error("CUDA error");
    }
    // Launch low-level kernel
    cuda<T>(stream, m, n, k, dev_src, dev_sumnorm);
    cuda_err = cudaStreamSynchronize(stream);
    if(cuda_err != cudaSuccess)
    {
        throw std::runtime_error("CUDA error");
    }
    // Copy result and deallocate device memory
    cuda_err = cudaMemcpy(&sumnorm[0], dev_sumnorm, sizeof(T)*2*m*n,
            cudaMemcpyDeviceToHost);
    if(cuda_err != cudaSuccess)
    {
        throw std::runtime_error("CUDA error");
    }
    cuda_err = cudaFree(dev_src);
    if(cuda_err != cudaSuccess)
    {
        throw std::runtime_error("CUDA error");
    }
    cuda_err = cudaFree(dev_sumnorm);
    if(cuda_err != cudaSuccess)
    {
        throw std::runtime_error("CUDA error");
    }
    cuda_err = cudaStreamDestroy(stream);
    if(cuda_err != cudaSuccess)
    {
        throw std::runtime_error("CUDA error");
    }
}
#endif // NNTILE_USE_CUDA

// Templated validation
template<typename T>
void validate(Index m, Index n, Index k)
{
    // Init test input
    std::vector<T> src(m*n*k), sumnorm(2*m*n);
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
    std::vector<T> sumnorm_copy(sumnorm);
    // Check low-level kernel
    std::cout << "Run kernel::sumnorm::cpu<T>\n";
    cpu<T>(m, n, k, &src[0], &sumnorm[0]);
    for(Index i0 = 0; i0 < m; ++i0)
    {
        for(Index i1 = 0; i1 < n; ++i1)
        {
            Index a = i0 + i1;
            T sum_ref = k * (2*a+k-1) / 2 / T{10};
            T sum = sumnorm[2*(i1*m+i0)];
            if(std::abs(sum/sum_ref-T{1}) / std::numeric_limits<T>::epsilon()
                    > 10)
            {
                throw std::runtime_error("Wrong sum");
            }
            T norm_sqr_ref = k * (2*(a-1)*a+(2*a+k-1)*(2*a+2*k-1)) / 6
                / T{100};
            T norm = sumnorm[2*(i1*m+i0)+1];
            T norm_sqr = norm * norm;
            if(std::abs(norm_sqr/norm_sqr_ref-T{1})
                    / std::numeric_limits<T>::epsilon() > 10)
            {
                throw std::runtime_error("Wrong norm");
            }
        }
    }
    std::cout << "OK: kernel::sumnorm::cpu<T>\n";
#ifdef NNTILE_USE_CUDA
    // Check low-level CUDA kernel
    sumnorm = sumnorm_copy;
    std::cout << "Run kernel::sumnorm::cuda<T>\n";
    run_cuda<T>(m, n, k, src, sumnorm);
    for(Index i0 = 0; i0 < m; ++i0)
    {
        for(Index i1 = 0; i1 < n; ++i1)
        {
            Index a = i0 + i1;
            T sum_ref = k * (2*a+k-1) / 2 / T{10};
            T sum = sumnorm[2*(i1*m+i0)];
            if(std::abs(sum/sum_ref-T{1}) / std::numeric_limits<T>::epsilon()
                    > 10)
            {
                throw std::runtime_error("Wrong sum");
            }
            T norm_sqr_ref = k * (2*(a-1)*a+(2*a+k-1)*(2*a+2*k-1)) / 6
                / T{100};
            T norm = sumnorm[2*(i1*m+i0)+1];
            T norm_sqr = norm * norm;
            if(std::abs(norm_sqr/norm_sqr_ref-T{1})
                    / std::numeric_limits<T>::epsilon() > 10)
            {
                throw std::runtime_error("Wrong norm");
            }
        }
    }
    std::cout << "OK: kernel::sumnorm::cuda<T>\n";
#endif // NNTILE_USE_CUDA
    // Check low-level kernel even more
    sumnorm_copy = sumnorm;
    std::cout << "Run kernel::sumnorm::cpu<T>\n";
    cpu<T>(m, n, k, &src[0], &sumnorm[0]);
    for(Index i0 = 0; i0 < m; ++i0)
    {
        for(Index i1 = 0; i1 < n; ++i1)
        {
            Index i = 2 * (i1*m+i0);
            if(std::abs(sumnorm[i]/sumnorm_copy[i]-T{2})
                    / std::numeric_limits<T>::epsilon() > 10)
            {
                throw std::runtime_error("Wrong sum");
            }
            if(std::abs(sumnorm[i+1]/sumnorm_copy[i+1]-std::sqrt(T{2}))
                    / std::numeric_limits<T>::epsilon() > 10)
            {
                throw std::runtime_error("Wrong norm");
            }
        }
    }
    std::cout << "OK: kernel::sumnorm::cpu<T>\n";
#ifdef NNTILE_USE_CUDA
    // Check low-level CUDA kernel
    sumnorm = sumnorm_copy;
    std::cout << "Run kernel::sumnorm::cuda<T>\n";
    run_cuda<T>(m, n, k, src, sumnorm);
    for(Index i0 = 0; i0 < m; ++i0)
    {
        for(Index i1 = 0; i1 < n; ++i1)
        {
            Index i = 2 * (i1*m+i0);
            if(std::abs(sumnorm[i]/sumnorm_copy[i]-T{2})
                    / std::numeric_limits<T>::epsilon() > 10)
            {
                throw std::runtime_error("Wrong sum");
            }
            if(std::abs(sumnorm[i+1]/sumnorm_copy[i+1]-std::sqrt(T{2}))
                    / std::numeric_limits<T>::epsilon() > 10)
            {
                throw std::runtime_error("Wrong norm");
            }
        }
    }
    std::cout << "OK: kernel::sumnorm::cuda<T>\n";
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

