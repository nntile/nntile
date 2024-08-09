/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/mask_scalar.cc
 * Mask scalar operation on a buffer
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/mask_scalar.hh"
#include "../testing.hh"
#include <vector>
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <memory>

using namespace nntile;
using namespace nntile::kernel::mask_scalar;

#ifdef NNTILE_USE_CUDA
template<typename T>
void run_cuda(Index nrows, Index ncols, const bool_t *mask, Scalar val,
        std::vector<T> &data)
{
    // Alloc on device
    T *dev_data;
    bool_t *dev_mask;
    Index nelems = nrows * ncols;
    cudaError_t cuda_err = cudaMalloc(&dev_data, sizeof(T)*nelems);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMalloc(&dev_mask, sizeof(bool_t)*nrows);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Copy to device
    cuda_err = cudaMemcpy(dev_data, &data[0], sizeof(T)*nelems,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_mask, mask, sizeof(bool_t)*nrows,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Init stream
    cudaStream_t stream;
    cuda_err = cudaStreamCreate(&stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Launch low-level kernel
    cuda<T>(stream, nrows, ncols, dev_mask, val, dev_data);
    cuda_err = cudaStreamSynchronize(stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Copy result and deallocate device memory
    cuda_err = cudaMemcpy(&data[0], dev_data, sizeof(T)*nelems,
            cudaMemcpyDeviceToHost);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaFree(dev_data);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaFree(dev_mask);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaStreamDestroy(stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
}
#endif // NNTILE_USE_CUDA

// Templated validation
template<typename T>
void validate(Index nrows, Index ncols)
{
    using Y = typename T::repr_t;
    Scalar val = -1.0;
    // Init test input
    Index nelems = nrows * ncols;
    std::vector<T> data(nelems);
    for(Index i = 0; i < nelems; ++i)
    {
        data[i] = Y(2*i+1-nelems) / Y{1000};
    }
    std::unique_ptr<bool_t[]> mask(new bool_t[nrows]);
    for(Index i = 0; i < nrows; ++i)
    {
        mask[i] = bool_t(false);
    }
    // Check low-level kernel
    std::cout << "Run kernel::mask_scalar::cpu<" << T::type_repr << ">\n";
    cpu<T>(nrows, ncols, &(mask[0]), val, &data[0]);
    for(Index i = 0; i < nelems; ++i)
    {
        TEST_ASSERT(Y(data[i]) == val);
    }
    std::cout << "OK: kernel::mask_scalar::cpu<" << T::type_repr << ">\n";
    for(Index i = 0; i < nrows; ++i)
    {
        if(i % 2 == 0)
        {
            mask[i] = bool_t(true);
        }
    }
    Scalar val_old = val;
    val = 10000.0;
    cpu<T>(nrows, ncols, &(mask[0]), val, &data[0]);
    for(Index i = 0; i < nrows; ++i)
    {
        for(Index j = 0; j < ncols; ++j)
        {
            if(i % 2 != 0)
            {
                TEST_ASSERT(Y(data[j*nrows+i]) == val);
            }
            else
            {
                TEST_ASSERT(Y(data[j*nrows+i]) == val_old);
            }
        }
    }
#ifdef NNTILE_USE_CUDA
    // Check low-level CUDA kernel
    std::cout << "Run kernel::mask_scalar::cuda<" << T::type_repr << ">\n";
    val = -val;
    run_cuda<T>(nrows, ncols, &mask[0], val, data);
    for(Index i = 0; i < nrows; ++i)
    {
        for(Index j = 0; j < ncols; ++j)
        {
            if(i % 2 != 0)
            {
                TEST_ASSERT(Y(data[j*nrows+i]) == val);
            }
            else
            {
                TEST_ASSERT(Y(data[j*nrows+i]) == val_old);
            }
        }
    }
    std::cout << "OK: kernel::mask_scalar::cuda<" << T::type_repr << ">\n";
#endif // NNTILE_USE_CUDA
}

int main(int argc, char **argv)
{
    validate<fp32_t>(1000, 100);
    validate<fp32_t>(10, 1);
    validate<fp32_t>(256, 64);
    validate<fp64_t>(1000, 100);
    validate<fp64_t>(10, 1);
    validate<fp64_t>(256, 64);
    return 0;
}
