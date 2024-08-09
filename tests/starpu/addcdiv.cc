/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/starpu/addcdiv.cc
 * Addcdiv operation on a StarPU buffer
 *
 * @version 1.1.0
 * */

#include "nntile/starpu/addcdiv.hh"
#include "nntile/kernel/addcdiv.hh"
#include "../testing.hh"
#ifdef NNTILE_USE_CUDA
#   include <cuda_runtime.h>
#endif // NNTILE_USE_CUDA
#include <vector>
#include <stdexcept>
#include <iostream>

using namespace nntile;
using namespace nntile::starpu;

template<typename T>
void validate_cpu(Scalar val, Scalar eps, Index nelems)
{
    using Y = typename T::repr_t;
    // Init all the data
    std::vector<T> data(nelems);
    std::vector<T> nom(nelems);
    std::vector<T> denom(nelems);
    for(Index i = 0; i < nelems; ++i)
    {
        data[i] = Y(i+1);
        nom[i] = Y(2*i + 1);
        denom[i] = Y(i);
    }
    // Create copies of destination
    std::vector<T> data2(data);
    // Launch low-level kernel
    std::cout << "Run kernel::addcdiv::cpu<" << T::type_repr << ">\n";
    kernel::addcdiv::cpu<T>(val, eps, nelems, &nom[0], &denom[0], &data[0]);
    // Check by actually submitting a task
    VariableHandle data2_handle(&data2[0], sizeof(T)*nelems, STARPU_RW);
    VariableHandle nom_handle(&nom[0], sizeof(T)*nelems, STARPU_R);
    VariableHandle denom_handle(&denom[0], sizeof(T)*nelems, STARPU_R);
    addcdiv::restrict_where(STARPU_CPU);
    std::cout << "Run starpu::addcdiv::submit<" << T::type_repr << "> restricted to CPU\n";
    addcdiv::submit<T>(val, eps, nelems, nom_handle, denom_handle, data2_handle);
    starpu_task_wait_for_all();
    data2_handle.unregister();
    // Check result
    for(Index i = 0; i < nelems; ++i)
    {
        TEST_ASSERT(Y(data[i]) == Y(data2[i]));
    }
    std::cout << "OK: starpu::addcdiv::submit<" << T::type_repr << "> restricted to CPU\n";
}

#ifdef NNTILE_USE_CUDA
// template<typename T>
// void validate_cuda(Index nelems)
// {
//     // Get a StarPU CUDA worker (to perform computations on the same device)
//     int cuda_worker_id = starpu_worker_get_by_type(STARPU_CUDA_WORKER, 0);
//     // Choose worker CUDA device
//     int dev_id = starpu_worker_get_devid(cuda_worker_id);
//     cudaError_t cuda_err = cudaSetDevice(dev_id);
//     TEST_ASSERT(cuda_err == cudaSuccess);
//     // Create CUDA stream
//     cudaStream_t stream;
//     cuda_err = cudaStreamCreate(&stream);
//     TEST_ASSERT(cuda_err == cudaSuccess);
//     // Init all the data
//     std::vector<T> data(nelems);
//     for(Index i = 0; i < nelems; ++i)
//     {
//         data[i] = T(i+1);
//     }
//     // Create copies of destination
//     std::vector<T> data2(data);
//     // Launch low-level kernel
//     T *dev_data;
//     cuda_err = cudaMalloc(&dev_data, sizeof(T)*nelems);
//     TEST_ASSERT(cuda_err == cudaSuccess);
//     cuda_err = cudaMemcpy(dev_data, &data[0], sizeof(T)*nelems,
//             cudaMemcpyHostToDevice);
//     TEST_ASSERT(cuda_err == cudaSuccess);
//     std::cout << "Run kernel::relu::cuda<T>\n";
//     kernel::relu::cuda<T>(stream, nelems, dev_data);
//     // Wait for result and destroy stream
//     cuda_err = cudaStreamSynchronize(stream);
//     TEST_ASSERT(cuda_err == cudaSuccess);
//     cuda_err = cudaStreamDestroy(stream);
//     TEST_ASSERT(cuda_err == cudaSuccess);
//     // Copy result back to CPU
//     cuda_err = cudaMemcpy(&data[0], dev_data, sizeof(T)*nelems,
//             cudaMemcpyDeviceToHost);
//     TEST_ASSERT(cuda_err == cudaSuccess);
//     // Deallocate CUDA memory
//     cuda_err = cudaFree(dev_data);
//     TEST_ASSERT(cuda_err == cudaSuccess);
//     // Check by actually submitting a task
//     VariableHandle data2_handle(&data2[0], sizeof(T)*nelems, STARPU_RW);
//     relu::restrict_where(STARPU_CUDA);
//     std::cout << "Run starpu::relu::submit<T> restricted to CUDA\n";
//     relu::submit<T>(nelems, data2_handle);
//     starpu_task_wait_for_all();
//     data2_handle.unregister();
//     // Check result
//     for(Index i = 0; i < nelems; ++i)
//     {
//         TEST_ASSERT(data[i] == data2[i]);
//     }
//     std::cout << "OK: starpu::relu::submit<T> restricted to CUDA\n";
// }
#endif // NNTILE_USE_CUDA

int main(int argc, char **argv)
{
    // Init StarPU for testing
    Config starpu(1, 1, 0);
    // Init codelet
    addcdiv::init();
    // Launch all tests
    validate_cpu<fp32_t>(1, 1e-3, 1);
    validate_cpu<fp32_t>(-10, 1e-4, 10000);

    validate_cpu<fp64_t>(1, 1e-8, 1);
    validate_cpu<fp64_t>(-10, 1e-6, 10000);
#ifdef NNTILE_USE_CUDA
    // validate_cuda<fp32_t>(1);
    // validate_cuda<fp32_t>(10000);
    // validate_cuda<fp64_t>(1);
    // validate_cuda<fp64_t>(10000);
#endif // NNTILE_USE_CUDA
    return 0;
}
