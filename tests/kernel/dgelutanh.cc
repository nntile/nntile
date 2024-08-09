/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/dgelutanh.cc
 * Derivative of approximate GeLU operation on a buffer
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/dgelutanh.hh"
#include "nntile/kernel/gelutanh_inplace.hh"
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
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_data, &data[0], sizeof(T)*nelems,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Init stream
    cudaStream_t stream;
    cuda_err = cudaStreamCreate(&stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Launch low-level CUDA kernel
    cuda<T>(stream, nelems, dev_data);
    cuda_err = cudaStreamSynchronize(stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Copy result and deallocate device memory
    cuda_err = cudaMemcpy(&data[0], dev_data, sizeof(T)*nelems,
            cudaMemcpyDeviceToHost);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaFree(dev_data);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaStreamDestroy(stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
}
#endif // NNTILE_USE_CUDA

// Templated validation
template<typename T>
void validate(Index nelems)
{
    using Y = typename T::repr_t;
    const Y eps = 1e-5;
    constexpr Y pi = 3.141592653589793238462643383279502884L;
    // Init test input
    std::vector<T> data(nelems);
    for(Index i = 0; i < nelems; ++i)
    {
        data[i] = Y(2*i+1-nelems) / Y{1000};
    }
    std::vector<T> data_save(data);
    // Check low-level CPU kernel
    std::cout << "Run kernel::dgelutanh::cpu<" << T::type_repr << ">\n";
    cpu<T>(nelems, &data[0]);
    for(Index i = 0; i < nelems; ++i)
    {
        // AGeLU'(z) = 0.5 + 0.5 tanh ( 0.0356774 z^3 + 0.797885 z) +
        // ( 0.0535161 z^3 + 0.398942 z) sech^2 ( 0.0356774 z^3 + 0.797885 z)
        Y x{data_save[i]};
        Y x1 = 0.0356774*x*x*x + 0.797885*x;
        Y x2 = 0.0535161*x*x*x + 0.398942*x;
        Y exp = std::exp(x1);
        Y val_ref;
        if(std::isinf(exp))
        {
            val_ref = Y{1};
        }
        else
        {
            Y cosh = std::cosh(x1);
            Y inv_cosh = Y{1} / cosh;
            val_ref = (Y{0.5}*exp+x2*inv_cosh) * inv_cosh;
        }
        // Obtain range of correct values
        Y val_ref_min, val_ref_max;
        if(val_ref < 0)
        {
            val_ref_min = val_ref * (Y{1}+eps) - eps;
            val_ref_max = val_ref * (Y{1}-eps) + eps;
        }
        else
        {
            val_ref_min = val_ref * (Y{1}-eps) - eps;
            val_ref_max = val_ref * (Y{1}+eps) + eps;
        }
        // NaN-aware comparisons
        TEST_ASSERT(Y(data[i]) >= val_ref_min and Y(data[i]) <= val_ref_max);
    }
    std::cout << "OK: kernel::dgelutanh::cpu<" << T::type_repr << ">\n";
#ifdef NNTILE_USE_CUDA
    // Check low-level CUDA kernel
    data = data_save;
    std::cout << "Run kernel::dgelutanh::cuda<" << T::type_repr << ">\n";
    run_cuda<T>(nelems, data);
    for(Index i = 0; i < nelems; ++i)
    {
        // AGeLU'(z) = 0.5 + 0.5 tanh ( 0.0356774 z^3 + 0.797885 z) +
        // ( 0.0535161 z^3 + 0.398942 z) sech^2 ( 0.0356774 z^3 + 0.797885 z)
        Y x{data_save[i]};
        Y x1 = 0.0356774*x*x*x + 0.797885*x;
        Y x2 = 0.0535161*x*x*x + 0.398942*x;
        Y exp = std::exp(x1);
        Y val_ref;
        if(std::isinf(exp))
        {
            val_ref = Y{1};
        }
        else
        {
            Y cosh = std::cosh(x1);
            Y inv_cosh = Y{1} / cosh;
            val_ref = (Y{0.5}*exp+x2*inv_cosh) * inv_cosh;
        }
        // Obtain range of correct values
        Y val_ref_min, val_ref_max;
        if(val_ref < 0)
        {
            val_ref_min = val_ref * (Y{1}+eps) - eps;
            val_ref_max = val_ref * (Y{1}-eps) + eps;
        }
        else
        {
            val_ref_min = val_ref * (Y{1}-eps) - eps;
            val_ref_max = val_ref * (Y{1}+eps) + eps;
        }
        // NaN-aware comparisons
        TEST_ASSERT(Y(data[i]) >= val_ref_min and Y(data[i]) <= val_ref_max);
    }
    std::cout << "OK: kernel::dgelutanh::cuda<" << T::type_repr << ">\n";
#endif // NNTILE_USE_CUDA
    // Check if dgelu is a derivative of gelu numerically
    std::vector<T> data2(data_save), data3(data_save);
    constexpr Y h = 1e-3, inv_h = 1/h;
    for(Index i = 0; i < nelems; ++i)
    {
        data2[i] = Y(data2[i]) + h/2;
        data3[i] = Y(data3[i]) - h/2;
    }
    gelutanh_inplace::cpu<T>(nelems, &data2[0]);
    gelutanh_inplace::cpu<T>(nelems, &data3[0]);
    for(Index i = 0; i < nelems; ++i)
    {
        Y num_x = inv_h * (Y(data2[i])-Y(data3[i]));
        Y diff = std::abs(num_x - Y(data[i]));
        Y abs = std::abs(Y(data[i]));
        Y threshold = abs * 4e-2;
        TEST_ASSERT(diff <= threshold
            or (diff > threshold and abs < eps));
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
