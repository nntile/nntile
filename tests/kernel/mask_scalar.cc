/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/mask_sclar.cc
 * Mask scalar operation on a buffer
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @author Aleksandr Mikhalev
 * @date 2023-06-27
 * */

#include "nntile/kernel/mask_scalar.hh"
#include "../testing.hh"
#include <vector>
#include <stdexcept>
#include <cmath>
#include <iostream>

using namespace nntile;
using namespace nntile::kernel::mask_scalar;

#ifdef NNTILE_USE_CUDA
// template<typename T>
// void run_cuda(Index nelems, T val, std::vector<T> &data)
// {
//     // Copy to device
//     T *dev_data;
//     cudaError_t cuda_err = cudaMalloc(&dev_data, sizeof(T)*nelems);
//     TEST_ASSERT(cuda_err == cudaSuccess);
//     cuda_err = cudaMemcpy(dev_data, &data[0], sizeof(T)*nelems,
//             cudaMemcpyHostToDevice);
//     TEST_ASSERT(cuda_err == cudaSuccess);
//     // Init stream
//     cudaStream_t stream;
//     cuda_err = cudaStreamCreate(&stream);
//     TEST_ASSERT(cuda_err == cudaSuccess);
//     // Launch low-level kernel
//     cuda<T>(stream, nelems, val, dev_data);
//     cuda_err = cudaStreamSynchronize(stream);
//     TEST_ASSERT(cuda_err == cudaSuccess);
//     // Copy result and deallocate device memory
//     cuda_err = cudaMemcpy(&data[0], dev_data, sizeof(T)*nelems,
//             cudaMemcpyDeviceToHost);
//     TEST_ASSERT(cuda_err == cudaSuccess);
//     cuda_err = cudaFree(dev_data);
//     TEST_ASSERT(cuda_err == cudaSuccess);
//     cuda_err = cudaStreamDestroy(stream);
//     TEST_ASSERT(cuda_err == cudaSuccess);
// }
#endif // NNTILE_USE_CUDA

// Templated validation
template<typename T>
void validate(Index nelems, Index batch_ndim)
{
    T val = -1.0;
    // Init test input
    std::vector<T> data(nelems);
    for(Index i = 0; i < nelems; ++i)
    {
        data[i] = T(2*i+1-nelems) / T{1000};
    }
    bool_t* mask = new bool_t[nelems/batch_ndim];
    for(Index i = 0; i < nelems/batch_ndim; ++i)
    {
        mask[i] = bool_t(false);
    }
    // Check low-level kernel
    std::cout << "Run kernel::mask_scalar::cpu<T>\n";
    cpu<T>(nelems, batch_ndim, mask, val, &data[0]);
    for(Index i = 0; i < nelems; ++i)
    {
        TEST_ASSERT(data[i] == val);
    }
    std::cout << "OK: kernel::mask_scalar::cpu<T>\n";

    for (Index k = 0; k < nelems/batch_ndim; ++k)
    {
        if (k % 2 == 0) 
        {
            mask[k] = true;
        }
    }
    cpu<T>(nelems, batch_ndim, mask, 10000, &data[0]);
    for(Index i = 0; i < nelems; ++i)
    {
        if (i % 2 != 0)
        {
            TEST_ASSERT(data[i] == 10000);
        }
    }
    delete[] mask;
#ifdef NNTILE_USE_CUDA
    // // Check low-level CUDA kernel
    // data = data_save;
    // std::cout << "Run kernel::mask_scalar::cuda<T>\n";
    // run_cuda<T>(nelems, val, data);
    // for(Index i = 0; i < nelems; ++i)
    // {
    //     TEST_ASSERT(data[i] == val);
    // }
    // std::cout << "OK: kernel::mask_scalar::cuda<T>\n";
#endif // NNTILE_USE_CUDA
}

int main(int argc, char **argv)
{
    validate<fp32_t>(100, 10);
    validate<fp32_t>(10, 1);
    validate<fp32_t>(256, 64);
    validate<fp64_t>(100, 10);
    validate<fp64_t>(10, 1);
    validate<fp64_t>(256, 64);
    return 0;
}