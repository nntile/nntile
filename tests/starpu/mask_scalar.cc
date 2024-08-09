/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/starpu/mask_scalar.cc
 * Mask scalar operation on a StarPU buffer
 *
 * @version 1.1.0
 * */

#include "nntile/starpu/mask_scalar.hh"
#include "nntile/kernel/mask_scalar.hh"
#include "../testing.hh"
#ifdef NNTILE_USE_CUDA
#   include <cuda_runtime.h>
#endif // NNTILE_USE_CUDA
#include <vector>
#include <stdexcept>
#include <iostream>
#include <memory>

using namespace nntile;
using namespace nntile::starpu;

template<typename T>
void validate_cpu(Index nrows, Index ncols)
{
    using Y = typename T::repr_t;
    // Init all the data
    Scalar val = -0.5;
    Index nelems = nrows * ncols;
    std::vector<T> data(nelems);
    for(Index i = 0; i < nelems; ++i)
    {
        data[i] = Y(i+1);
    }
    std::unique_ptr<bool_t[]> mask(new bool_t[nrows]);
    for(Index i = 0; i < nrows; ++i)
    {
        if(i % 2 == 0)
        {
            mask[i] = bool_t(false);
        }
        else
        {
            mask[i] = bool_t(true);
        }
    }
    // Create copies of destination
    std::vector<T> data2(data);
    // Launch low-level kernel
    std::cout << "Run kernel::mask_scalar::cpu<" << T::type_repr << ">\n";
    kernel::mask_scalar::cpu<T>(nrows, ncols, &mask[0], val, &data[0]);
    // Check by actually submitting a task
    VariableHandle data2_handle(&data2[0], sizeof(T)*nelems, STARPU_RW);
    VariableHandle mask_handle(&mask[0], sizeof(bool_t)*nrows, STARPU_RW);
    mask_scalar::restrict_where(STARPU_CPU);
    std::cout << "Run starpu::mask_scalar::submit<" << T::type_repr << "> restricted to CPU\n";
    mask_scalar::submit<T>(nrows, ncols, mask_handle, val, data2_handle);
    starpu_task_wait_for_all();
    data2_handle.unregister();
    mask_handle.unregister();
    // Check result
    for(Index i = 0; i < nelems; ++i)
    {
        TEST_ASSERT(Y(data[i]) == Y(data2[i]));
    }
    std::cout << "OK: starpu::mask_scalar::submit<" << T::type_repr << "> restricted to CPU\n";
}

#ifdef NNTILE_USE_CUDA
template<typename T>
void validate_cuda(Index nrows, Index ncols)
{
    using Y = typename T::repr_t;
    // Get a StarPU CUDA worker (to perform computations on the same device)
    int cuda_worker_id = starpu_worker_get_by_type(STARPU_CUDA_WORKER, 0);
    // Choose worker CUDA device
    int dev_id = starpu_worker_get_devid(cuda_worker_id);
    cudaError_t cuda_err = cudaSetDevice(dev_id);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Create CUDA stream
    cudaStream_t stream;
    cuda_err = cudaStreamCreate(&stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Init all the data
    Scalar val = -0.5;
    Index nelems = nrows * ncols;
    std::vector<T> data(nelems);
    for(Index i = 0; i < nelems; ++i)
    {
        data[i] = Y(i+1);
    }
    std::unique_ptr<bool_t[]> mask(new bool_t[nrows]);
    for(Index i = 0; i < nrows; ++i)
    {
        if(i % 2 == 0)
        {
            mask[i] = bool_t(false);
        }
        else
        {
            mask[i] = bool_t(true);
        }
    }
    // Create copies of destination
    std::vector<T> data2(data);
    // Launch low-level kernel
    T *dev_data;
    bool_t *dev_mask;
    cuda_err = cudaMalloc(&dev_data, sizeof(T)*nelems);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMalloc(&dev_mask, sizeof(bool_t)*nrows);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_data, &data[0], sizeof(T)*nelems,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_mask, &mask[0], sizeof(bool_t)*nrows,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    std::cout << "Run kernel::mask_scalar::cuda<" << T::type_repr << ">\n";
    kernel::mask_scalar::cuda<T>(stream, nrows, ncols, dev_mask, val,
            dev_data);
    // Wait for result and destroy stream
    cuda_err = cudaStreamSynchronize(stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaStreamDestroy(stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Copy result back to CPU
    cuda_err = cudaMemcpy(&data[0], dev_data, sizeof(T)*nelems,
            cudaMemcpyDeviceToHost);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Deallocate CUDA memory
    cuda_err = cudaFree(dev_data);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaFree(dev_mask);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Check by actually submitting a task
    VariableHandle data2_handle(&data2[0], sizeof(T)*nelems, STARPU_RW);
    VariableHandle mask_handle(&mask[0], sizeof(bool_t)*nrows, STARPU_RW);
    mask_scalar::restrict_where(STARPU_CUDA);
    std::cout << "Run starpu::mask_scalar::submit<" << T::type_repr << "> restricted to CUDA\n";
    mask_scalar::submit<T>(nrows, ncols, mask_handle, val, data2_handle);
    starpu_task_wait_for_all();
    data2_handle.unregister();
    mask_handle.unregister();
    // Check result
    for(Index i = 0; i < nelems; ++i)
    {
        TEST_ASSERT(Y(data[i]) == Y(data2[i]));
    }
    std::cout << "OK: starpu::mask_scalar::submit<" << T::type_repr << "> restricted to CUDA\n";
}
#endif // NNTILE_USE_CUDA

int main(int argc, char **argv)
{
    // Init StarPU for testing
    Config starpu(1, 1, 0);
    // Init codelet
    mask_scalar::init();
    // Launch all tests
    validate_cpu<fp32_t>(1000, 10);
    validate_cpu<fp32_t>(10, 1);
    validate_cpu<fp32_t>(81, 3);
    validate_cpu<fp64_t>(1000, 10);
    validate_cpu<fp64_t>(10, 1);
    validate_cpu<fp64_t>(81, 3);
#ifdef NNTILE_USE_CUDA
    validate_cuda<fp32_t>(1, 2);
    validate_cuda<fp32_t>(10000, 10);
    validate_cuda<fp64_t>(1, 2);
    validate_cuda<fp64_t>(10000, 10);
#endif // NNTILE_USE_CUDA
    return 0;
}
