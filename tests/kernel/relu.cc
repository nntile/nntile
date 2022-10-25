/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/relu.cc
 * ReLU operation on a buffer
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-10-25
 * */

#include "nntile/kernel/relu.hh"
#include "../testing.hh"
#include <vector>
#include <stdexcept>
#include <cmath>
#include <iostream>

using namespace nntile;
using namespace nntile::kernel::relu;

#ifdef NNTILE_USE_CUDA
template<typename T>
void run_cuda(Index nelems, std::vector<T> &data)
{
    // Copy to device
    T *dev_data;
    cudaError_t cuda_err = cudaMalloc(&dev_data, sizeof(T)*nelems);
    TEST_ASSERT(cuda_error == cudaSuccess);
    cuda_err = cudaMemcpy(dev_data, &data[0], sizeof(T)*nelems,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_error == cudaSuccess);
    // Init stream
    cudaStream_t stream;
    cuda_err = cudaStreamCreate(&stream);
    TEST_ASSERT(cuda_error == cudaSuccess);
    // Launch low-level kernel
    cuda<T>(stream, nelems, dev_data);
    cuda_err = cudaStreamSynchronize(stream);
    TEST_ASSERT(cuda_error == cudaSuccess);
    // Copy result and deallocate device memory
    cuda_err = cudaMemcpy(&data[0], dev_data, sizeof(T)*nelems,
            cudaMemcpyDeviceToHost);
    TEST_ASSERT(cuda_error == cudaSuccess);
    cuda_err = cudaFree(dev_data);
    TEST_ASSERT(cuda_error == cudaSuccess);
    cuda_err = cudaStreamDestroy(stream);
    TEST_ASSERT(cuda_error == cudaSuccess);
}
#endif // NNTILE_USE_CUDA

// Templated validation
template<typename T>
void validate(Index nelems)
{
    constexpr T eps = std::numeric_limits<T>::epsilon();
    // Init test input
    std::vector<T> data(nelems);
    for(Index i = 0; i < nelems; ++i)
    {
        data[i] = T(2*i+1-nelems) / T{1000};
    }
    std::vector<T> data_save(data);
    // Check low-level kernel
    std::cout << "Run kernel::relu::cpu<T>\n";
    cpu<T>(nelems, &data[0]);
    for(Index i = 0; i < nelems; ++i)
    {
        T x = data_save[i];
        T val_ref = std::max(x, T{0});
        TEST_ASSERT(data[i] == val_ref);
    }
    std::cout << "OK: kernel::relu::cpu<T>\n";
#ifdef NNTILE_USE_CUDA
    // Check low-level CUDA kernel
    data = data_save;
    std::cout << "Run kernel::relu::cuda<T>\n";
    run_cuda<T>(nelems, data);
    for(Index i = 0; i < nelems; ++i)
    {
        T x = data_save[i];
        T val_ref = std::max(x, T{0});
        TEST_ASSERT(data[i] == val_ref);
    }
    std::cout << "OK: kernel::relu::cuda<T>\n";
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

