/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/starpu/prod_inplace.cc
 * Per-element product of two StarPU buffers
 *
 * @version 1.1.0
 * */

#include "nntile/starpu/prod_inplace.hh"
#include "nntile/kernel/prod_inplace.hh"
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
void validate_cpu(Index nelems)
{
    using Y = typename T::repr_t;
    // Init all the data
    std::vector<T> src(nelems), dst(nelems);
    for(Index i = 0; i < nelems; ++i)
    {
        src[i] = Y(2*i+1-nelems) / Y{1000};
        dst[i] = Y(nelems-i) / Y{1000};
    }
    // Create copies of destination
    std::vector<T> dst2(dst);
    // Launch low-level kernel
    std::cout << "Run kernel::prod_inplace::cpu<" << T::type_repr << ">\n";
    kernel::prod_inplace::cpu<T>(nelems, &src[0], &dst[0]);
    // Check by actually submitting a task
    VariableHandle src_handle(&src[0], sizeof(T)*nelems, STARPU_R),
        dst2_handle(&dst2[0], sizeof(T)*nelems, STARPU_RW);
    prod_inplace::restrict_where(STARPU_CPU);
    std::cout << "Run starpu::prod_inplace::submit<" << T::type_repr << "> restricted to CPU\n";
    prod_inplace::submit<T>(nelems, src_handle, dst2_handle);
    starpu_task_wait_for_all();
    src_handle.unregister();
    dst2_handle.unregister();
    // Check result
    for(Index i = 0; i < nelems; ++i)
    {
        TEST_ASSERT(Y(dst[i]) == Y(dst2[i]));
    }
    std::cout << "OK: starpu::prod_inplace::submit<" << T::type_repr << "> restricted to CPU\n";
}

#ifdef NNTILE_USE_CUDA
template<typename T>
void validate_cuda(Index nelems)
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
    std::vector<T> src(nelems), dst(nelems);
    for(Index i = 0; i < nelems; ++i)
    {
        src[i] = Y(2*i+1-nelems) / Y{1000};
        dst[i] = Y(nelems-i) / Y{1000};
    }
    // Create copies of destination
    std::vector<T> dst2(dst);
    // Launch low-level kernel
    T *dev_src, *dev_dst;
    cuda_err = cudaMalloc(&dev_src, sizeof(T)*nelems);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMalloc(&dev_dst, sizeof(T)*nelems);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_src, &src[0], sizeof(T)*nelems,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_dst, &dst[0], sizeof(T)*nelems,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    std::cout << "Run kernel::prod_inplace::cuda<" << T::type_repr << ">\n";
    kernel::prod_inplace::cuda<T>(stream, nelems, dev_src, dev_dst);
    // Wait for result and destroy stream
    cuda_err = cudaStreamSynchronize(stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaStreamDestroy(stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Copy result back to CPU
    cuda_err = cudaMemcpy(&dst[0], dev_dst, sizeof(T)*nelems,
            cudaMemcpyDeviceToHost);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Deallocate CUDA memory
    cuda_err = cudaFree(dev_src);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaFree(dev_dst);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Check by actually submitting a task
    VariableHandle src_handle(&src[0], sizeof(T)*nelems, STARPU_R),
        dst2_handle(&dst2[0], sizeof(T)*nelems, STARPU_RW);
    prod_inplace::restrict_where(STARPU_CUDA);
    std::cout << "Run starpu::prod_inplace::submit<" << T::type_repr << "> restricted to CUDA\n";
    prod_inplace::submit<T>(nelems, src_handle, dst2_handle);
    starpu_task_wait_for_all();
    src_handle.unregister();
    dst2_handle.unregister();
    // Check result
    for(Index i = 0; i < nelems; ++i)
    {
        TEST_ASSERT(Y(dst[i]) == Y(dst2[i]));
    }
    std::cout << "OK: starpu::prod_inplace::submit<" << T::type_repr << "> restricted to CUDA\n";
}
#endif // NNTILE_USE_CUDA

int main(int argc, char **argv)
{
    // Init StarPU for testing
    Config starpu(1, 1, 0);
    // Init codelet
    prod_inplace::init();
    // Launch all tests
    validate_cpu<fp32_t>(1);
    validate_cpu<fp32_t>(10000);
    validate_cpu<fp64_t>(1);
    validate_cpu<fp64_t>(10000);
#ifdef NNTILE_USE_CUDA
    validate_cuda<fp32_t>(1);
    validate_cuda<fp32_t>(10000);
    validate_cuda<fp64_t>(1);
    validate_cuda<fp64_t>(10000);
#endif // NNTILE_USE_CUDA
    return 0;
}
