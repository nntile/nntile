/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/dgelutanh.cc
 * Derivative of approximate GeLU operation on a buffer
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-10-25
 * */

#include "nntile/kernel/dgelutanh.hh"
#include "nntile/kernel/gelutanh.hh"
#include "../testing.hh"
#include <vector>
#include <stdexcept>
#include <cmath>
#include <iostream>

using namespace nntile;
using namespace nntile::kernel;
using namespace nntile::kernel::dgelutanh;

#ifdef NNTILE_USE_CUDA
template<typename T>
void run_cuda(Index nelems, std::vector<T> &data)
{
    // Copy to device
    T *dev_data;
    cudaError_t cuda_err = cudaMalloc(&dev_data, sizeof(T)*nelems);
    if(cuda_err != cudaSuccess)
    {
        throw std::runtime_error("CUDA error");
    }
    cuda_err = cudaMemcpy(dev_data, &data[0], sizeof(T)*nelems,
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
    // Launch low-level CUDA kernel
    cuda<T>(stream, nelems, dev_data);
    cuda_err = cudaStreamSynchronize(stream);
    if(cuda_err != cudaSuccess)
    {
        throw std::runtime_error("CUDA error");
    }
    // Copy result and deallocate device memory
    cuda_err = cudaMemcpy(&data[0], dev_data, sizeof(T)*nelems,
            cudaMemcpyDeviceToHost);
    if(cuda_err != cudaSuccess)
    {
        throw std::runtime_error("CUDA error");
    }
    cuda_err = cudaFree(dev_data);
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
void validate(Index nelems)
{
    constexpr T eps = std::max(std::numeric_limits<T>::epsilon(), T{1e-5f});
    constexpr T pi = 3.141592653589793238462643383279502884L;
    // Init test input
    std::vector<T> data(nelems);
    for(Index i = 0; i < nelems; ++i)
    {
        data[i] = T(2*i+1-nelems) / T{1000};
    }
    std::vector<T> data_save(data);
    // Check low-level CPU kernel
    std::cout << "Run kernel::dgelutanh::cpu<T>\n";
    cpu<T>(nelems, &data[0]);
    for(Index i = 0; i < nelems; ++i)
    {
        // AGeLU'(z) = 0.5 + 0.5 tanh ( 0.0356774 z^3 + 0.797885 z) +
        // ( 0.0535161 z^3 + 0.398942 z) sech^2 ( 0.0356774 z^3 + 0.797885 z)
        T x = data_save[i];
        T x1 = 0.0356774*x*x*x + 0.797885*x;
        T x2 = 0.0535161*x*x*x + 0.398942*x;
        T exp = std::exp(x1);
        T val_ref;
        if(std::isinf(exp))
        {
            val_ref = T{1};
        }
        else
        {
            T cosh = std::cosh(x1);
            T inv_cosh = T{1} / cosh;
            val_ref = (T{0.5}*exp+x2*inv_cosh) * inv_cosh;
        }
        // Obtain range of correct values
        T val_ref_min, val_ref_max;
        if(val_ref < 0)
        {
            val_ref_min = val_ref * (T{1}+eps) - eps;
            val_ref_max = val_ref * (T{1}-eps) + eps;
        }
        else
        {
            val_ref_min = val_ref * (T{1}-eps) - eps;
            val_ref_max = val_ref * (T{1}+eps) + eps;
        }
        if(data[i] < val_ref_min or data[i] > val_ref_max)
        {
            throw std::runtime_error("Wrong value");
        }
    }
    std::cout << "OK: kernel::dgelutanh::cpu<T>\n";
#ifdef NNTILE_USE_CUDA
    // Check low-level CUDA kernel
    data = data_save;
    std::cout << "Run kernel::dgelutanh::cuda<T>\n";
    run_cuda<T>(nelems, data);
    for(Index i = 0; i < nelems; ++i)
    {
        // AGeLU'(z) = 0.5 + 0.5 tanh ( 0.0356774 z^3 + 0.797885 z) +
        // ( 0.0535161 z^3 + 0.398942 z) sech^2 ( 0.0356774 z^3 + 0.797885 z)
        T x = data_save[i];
        T x1 = 0.0356774*x*x*x + 0.797885*x;
        T x2 = 0.0535161*x*x*x + 0.398942*x;
        T tanh = std::tanh(x1);
        T cosh = std::cosh(x1);
        T inv_cosh_2 = T{1} / (cosh*cosh);
        T val_ref = 0.5 + 0.5*tanh + x2*inv_cosh_2;
        // Obtain range of correct values
        T val_ref_min, val_ref_max;
        if(val_ref < 0)
        {
            val_ref_min = val_ref * (T{1}+eps) - eps;
            val_ref_max = val_ref * (T{1}-eps) + eps;
        }
        else
        {
            val_ref_min = val_ref * (T{1}-eps) - eps;
            val_ref_max = val_ref * (T{1}+eps) + eps;
        }
        if(data[i] < val_ref_min or data[i] > val_ref_max)
        {
            throw std::runtime_error("Wrong value");
        }
    }
    std::cout << "OK: kernel::dgelutanh::cuda<T>\n";
#endif // NNTILE_USE_CUDA
    return;
    // Check if dgelu is a derivative of gelu numerically
    std::vector<T> data2(data_save), data3(data_save);
    constexpr T h = 1e-3, inv_h = 1/h;
    for(Index i = 0; i < nelems; ++i)
    {
        data2[i] += h/2;
        data3[i] -= h/2;
    }
    gelutanh::cpu<T>(nelems, &data2[0]);
    gelutanh::cpu<T>(nelems, &data3[0]);
    for(Index i = 0; i < nelems; ++i)
    {
        T num_x = inv_h * (data2[i]-data3[i]);
        T diff = std::abs(num_x - data[i]);
        T abs = std::abs(data[i]);
        T threshold = abs * 5e-2;
        TEST_ASSERT(diff <= threshold or (diff > threshold and abs < 1e-4));
    }
}

int main(int argc, char **argv)
{
    validate<fp32_t>(0);
    validate<fp32_t>(1);
    validate<fp32_t>(80000);
    validate<fp64_t>(0);
    validate<fp64_t>(1);
    validate<fp64_t>(80000);
    return 0;
}

