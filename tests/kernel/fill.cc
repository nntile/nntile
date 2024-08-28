/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/fill.cc
 * Fill operation on a buffer
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/fill.hh"
#include "../testing.hh"
#include <vector>
#include <stdexcept>
#include <cmath>
#include <iostream>

using namespace nntile;
using namespace nntile::kernel::fill;

#ifdef NNTILE_USE_CUDA
template<typename T>
void run_cuda(Index nelems, Scalar val, std::vector<T> &data)
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
    // Launch low-level kernel
    cuda<T>(stream, nelems, val, dev_data);
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
    const Y eps = T::epsilon();
    Scalar val = -1.0;
    // Init test input
    std::vector<T> data(nelems);
    for(Index i = 0; i < nelems; ++i)
    {
        data[i] = Y(2*i+1-nelems) / Y{1000};
    }
    std::vector<T> data_save(data);
    // Check low-level kernel
    std::cout << "Run kernel::fill::cpu<" << T::type_repr << ">\n";
    cpu<T>(nelems, val, &data[0]);
    for(Index i = 0; i < nelems; ++i)
    {
        TEST_ASSERT(Y(data[i]) == val);
    }
    std::cout << "OK: kernel::fill::cpu<" << T::type_repr << ">\n";
#ifdef NNTILE_USE_CUDA
    // Check low-level CUDA kernel
    data = data_save;
    std::cout << "Run kernel::fill::cuda<" << T::type_repr << ">\n";
    run_cuda<T>(nelems, val, data);
    for(Index i = 0; i < nelems; ++i)
    {
        TEST_ASSERT(Y(data[i]) == val);
    }
    std::cout << "OK: kernel::fill::cuda<" << T::type_repr << ">\n";
#endif // NNTILE_USE_CUDA
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
