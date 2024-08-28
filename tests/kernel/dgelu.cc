/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/dgelu.cc
 * Derivative of GeLU operation on a buffer
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/dgelu.hh"
#include "nntile/kernel/gelu.hh"
#include "../testing.hh"
#include <vector>
#include <stdexcept>
#include <cmath>
#include <iostream>

using namespace nntile;
using namespace nntile::kernel;
using namespace nntile::kernel::dgelu;

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
    TEST_ASSERT(cuda_err == cudaSuccess)
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
    const Y eps = 2 * T::epsilon();
    constexpr Y pi = 3.141592653589793238462643383279502884L;
    // Init test input
    std::vector<T> data(nelems);
    for(Index i = 0; i < nelems; ++i)
    {
        data[i] = Y(2*i+1-nelems) / Y{1000};
    }
    std::vector<T> data_save(data);
    // Check low-level CPU kernel
    std::cout << "Run kernel::dgelu::cpu<" << T::type_repr << ">\n";
    cpu<T>(nelems, &data[0]);
    for(Index i = 0; i < nelems; ++i)
    {
        Y x(data_save[i]);
        Y val_ref = 0.5 * std::erfc(-x/std::sqrt(Y(2)));
        val_ref += x / std::sqrt(2*pi) * std::exp(-0.5*x*x);
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
    std::cout << "OK: kernel::dgelu::cpu<" << T::type_repr << ">\n";
#ifdef NNTILE_USE_CUDA
    // Check low-level CUDA kernel
    data = data_save;
    std::cout << "Run kernel::dgelu::cuda<" << T::type_repr << ">\n";
    run_cuda<T>(nelems, data);
    for(Index i = 0; i < nelems; ++i)
    {
        Y x(data_save[i]);
        Y val_ref = 0.5 * std::erfc(-x/std::sqrt(Y(2)));
        val_ref += x / std::sqrt(2*pi) * std::exp(-0.5*x*x);
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
    std::cout << "OK: kernel::dgelu::cuda<" << T::type_repr << ">\n";
#endif // NNTILE_USE_CUDA
    // Check if dgelu is a derivative of gelu numerically
    std::vector<T> data2(data_save), data3(data_save);
    constexpr Y h = 1e-3, inv_h = 1/h;
    for(Index i = 0; i < nelems; ++i)
    {
        data2[i] = Y(data2[i]) + h/2;
        data3[i] = Y(data3[i]) - h/2;
    }
    gelu::cpu<T>(nelems, &data2[0]);
    gelu::cpu<T>(nelems, &data3[0]);
    for(Index i = 0; i < nelems; ++i)
    {
        Y num_x = inv_h * (Y(data2[i])-Y(data3[i]));
        Y diff = std::abs(num_x - Y(data[i]));
        Y abs = std::abs(Y(data[i]));
        Y threshold = abs * 5e-2;
        // NaN-aware comparisons
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
