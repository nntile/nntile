/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/normalize.cc
 * Normalize operation for a buffer on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-15
 * */

#include "nntile/kernel/cpu/normalize.hh"
#include "nntile/defs.h"
#ifdef NNTILE_USE_CUDA
#   include "nntile/kernel/cuda/normalize.hh"
#endif // NNTILE_USE_CUDA
#include <vector>
#include <stdexcept>
#include <limits>
#include <cmath>

using namespace nntile;
using namespace nntile::kernel;

#ifdef NNTILE_USE_CUDA
template<typename T>
void run_cuda(Index m, Index n, Index k, Index l, T eps, T gamma, T beta,
        const std::vector<T> &sumnorm, std::vector<T> &dst);
{
    // Copy to device
    T *dev_sumnorm, *dev_dst;
    cudaError_t cuda_err = cudaMalloc(&dev_sumnorm, sizeof(T)*2*m*n);
    if(cuda_err != cudaSuccess)
    {
        throw std::runtime_error("CUDA error");
    }
    cuda_err = cudaMalloc(&dev_dst, sizeof(T)*m*n*k);
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
    cuda_err = cudaMemcpy(dev_dst, &dst[0], sizeof(T)*m*n*k,
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
    cuda::normalize<T>(stream, m, n, k, l, eps, gamma, beta, dev_sumnorm,
            dev_dst);
    cuda_err = cudaStreamSynchronize(stream);
    if(cuda_err != cudaSuccess)
    {
        throw std::runtime_error("CUDA error");
    }
    // Copy result and deallocate device memory
    cuda_err = cudaMemcpy(&dst[0], dev_dst, sizeof(T)*m*n*k,
            cudaMemcpyDeviceToHost);
    if(cuda_err != cudaSuccess)
    {
        throw std::runtime_error("CUDA error");
    }
    cuda_err = cudaFree(dev_sumnorm);
    if(cuda_err != cudaSuccess)
    {
        throw std::runtime_error("CUDA error");
    }
    cuda_err = cudaFree(dev_dst);
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
void validate(Index m, Index n, Index k, Index l, T eps, T gamma, T beta)
{
    // Init test input
    std::vector<T> sumnorm(2*m*n), dst(m*n*k);
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
            T avg = T(i0+i1) / T{10};
            sumnorm[2*(i1*m+i0)] = avg * T(l);
            sumnorm[2*(i1*m+i0)+1] = std::sqrt((avg*avg+T{1}+eps) * T(l));
        }
    }
    std::vector<T> dst_save(dst);
    // Check low-level kernel
    cpu::normalize<T>(m, n, k, l, eps, gamma, beta, &sumnorm[0], &dst[0]);
    for(Index i0 = 0; i0 < m; ++i0)
    {
        for(Index i1 = 0; i1 < n; ++i1)
        {
            for(Index i2 = 0; i2 < k; ++i2)
            {
                T val = dst[(i1*k+i2)*m+i0];
                T val_ref = T(i2)/T{10}/std::sqrt(T{1}+2*eps)*gamma + beta;
                if(std::abs(val/val_ref-T{1})
                        / std::numeric_limits<T>::epsilon() > 50)
                {
                    throw std::runtime_error("Wrong value");
                }
            }
        }
    }
#ifdef NNTILE_USE_CUDA
    // Check low-level CUDA kernel
    dst = dst_save;
    run_cuda<T>(m, n, k, l, eps, gamma, beta, sumnorm, dst);
    for(Index i0 = 0; i0 < m; ++i0)
    {
        for(Index i1 = 0; i1 < n; ++i1)
        {
            for(Index i2 = 0; i2 < k; ++i2)
            {
                T val = dst[(i1*k+i2)*m+i0];
                T val_ref = T(i2)/T{10}/std::sqrt(T{1}+2*eps)*gamma + beta;
                if(std::abs(val/val_ref-T{1})
                        / std::numeric_limits<T>::epsilon() > 50)
                {
                    throw std::runtime_error("Wrong value");
                }
            }
        }
    }
#endif // NNTILE_USE_CUDA
}

int main(int argc, char **argv)
{
    fp64_t eps[3] = {0.0, 1.0, 11.1};
    fp64_t gamma[3] = {0.0, 1.0, 3.3};
    fp64_t beta[3] = {0.0, 1.1, -2.2};
    for(Index i = 0 ; i < sizeof(eps)/sizeof(eps[0]); ++i)
    {
        for(Index j = 0 ; j < sizeof(gamma)/sizeof(gamma[0]); ++j)
        {
            for(Index k = 0 ; k < sizeof(beta)/sizeof(beta[0]); ++k)
            {
                validate<fp32_t>(1, 9, 11, 22, eps[i], gamma[j], beta[k]);
                validate<fp32_t>(8, 1, 11, 22, eps[i], gamma[j], beta[k]);
                validate<fp32_t>(8, 9, 1, 22, eps[i], gamma[j], beta[k]);
                validate<fp64_t>(1, 9, 11, 22, eps[i], gamma[j], beta[k]);
                validate<fp64_t>(8, 1, 11, 22, eps[i], gamma[j], beta[k]);
                validate<fp64_t>(8, 9, 1, 22, eps[i], gamma[j], beta[k]);
            }
        }
    }
    return 0;
}

