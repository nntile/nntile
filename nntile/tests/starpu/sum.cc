/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/starpu/sum.cc
 * Sum all elements of a StarPU buffer
 *
 * @version 1.1.0
 * */

#include "nntile/context.hh"
#include "nntile/starpu/sum.hh"
#include "nntile/kernel/sum.hh"
#include "../testing.hh"
#ifdef NNTILE_USE_CUDA
#   include <cuda_runtime.h>
#endif // NNTILE_USE_CUDA
#include <vector>
#include <stdexcept>
#include <cmath>
#include <iostream>

using namespace nntile;
using namespace nntile::starpu;

template<typename T>
void validate_cpu(Index nelems, Scalar alpha, Scalar beta)
{
    using Y = typename T::repr_t;
    // Init all the data
    std::vector<T> src(nelems);
    for(Index i = 0; i < nelems; ++i)
    {
        src[i] = Y(i+1);
    }
    std::vector<T> sum_dst(1);
    sum_dst[0] = Y(-1);
    // Create copies of destination
    std::vector<T> sum_dst2(sum_dst);
    // Launch low-level kernel
    std::cout << "Run kernel::sum::cpu<" << T::short_name << ">\n";
    kernel::sum::cpu<T>(nelems, alpha, &src[0], beta, &sum_dst[0]);
    // Check by actually submitting a task
    VariableHandle src_handle(&src[0], sizeof(T)*nelems),
        sum_dst2_handle(&sum_dst2[0], sizeof(T));
    sum.restrict_where(STARPU_CPU);
    std::cout << "Run starpu::sum::submit<" << T::short_name << "> restricted to CPU\n";
    sum.submit<std::tuple<T>>(nelems, alpha, src_handle, beta, sum_dst2_handle);
    starpu_task_wait_for_all();
    sum_dst2_handle.unregister();
    // Check result
    TEST_ASSERT(Y(sum_dst[0]) == Y(sum_dst2[0]));
    std::cout << "OK: starpu::sum::submit<" << T::short_name << "> restricted to CPU\n";
}

#ifdef NNTILE_USE_CUDA
template<typename T>
void validate_cuda(Index nelems, Scalar alpha, Scalar beta)
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
    std::vector<T> src(nelems);
    for(Index i = 0; i < nelems; ++i)
    {
        src[i] = Y(i+1);
    }
    std::vector<T> sum_dst(1);
    sum_dst[0] = Y(-1);
    // Create copies of destination
    std::vector<T> sum_dst2(sum_dst);
    // Launch low-level kernel
    T *dev_src, *dev_dst;
    cuda_err = cudaMalloc(&dev_src, sizeof(T)*nelems);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMalloc(&dev_dst, sizeof(T));
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_src, &src[0], sizeof(T)*nelems,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_dst, &sum_dst[0], sizeof(T),
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    std::cout << "Run kernel::sum::cuda<" << T::short_name << ">\n";
    kernel::sum::cuda<T>(stream, nelems, alpha, dev_src, beta, dev_dst);
    // Wait for result and destroy stream
    cuda_err = cudaStreamSynchronize(stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaStreamDestroy(stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Copy result back to CPU
    cuda_err = cudaMemcpy(&sum_dst[0], dev_dst, sizeof(T),
            cudaMemcpyDeviceToHost);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Deallocate CUDA memory
    cuda_err = cudaFree(dev_src);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaFree(dev_dst);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Check by actually submitting a task
    VariableHandle src_handle(&src[0], sizeof(T)*nelems),
        sum_dst2_handle(&sum_dst2[0], sizeof(T));
    sum.restrict_where(STARPU_CUDA);
    std::cout << "Run starpu::sum::submit<" << T::short_name << "> restricted to CUDA\n";
    sum.submit<std::tuple<T>>(nelems, alpha, src_handle, beta, sum_dst2_handle);
    starpu_task_wait_for_all();
    sum_dst2_handle.unregister();
    // Check result
    TEST_ASSERT(Y(sum_dst[0]) == Y(sum_dst2[0]));
    std::cout << "OK: starpu::sum::submit<" << T::short_name << "> restricted to CUDA\n";
}
#endif // NNTILE_USE_CUDA

int main(int argc, char **argv)
{
    // Initialize StarPU (it will automatically shutdown itself on exit)
    int ncpu=1, ncuda=1, ooc=0, verbose=0;
    const char *ooc_path = "/tmp/nntile_ooc";
    size_t ooc_size = 16777216;
    auto context = Context(ncpu, ncuda, ooc, ooc_path, ooc_size, verbose);

    // Launch all tests
    validate_cpu<fp32_t>(35, 1.0, -1.0);
    validate_cpu<fp32_t>(35, 2.0, 0.0);
    validate_cpu<fp32_t>(35, 0.0, 1.0);
    validate_cpu<fp64_t>(35, 1.0, -1.0);
    validate_cpu<fp64_t>(35, 2.0, 0.0);
    validate_cpu<fp64_t>(35, 0.0, 1.0);
#ifdef NNTILE_USE_CUDA
    validate_cuda<fp32_t>(35, 1.0, -1.0);
    validate_cuda<fp32_t>(35, 2.0, 0.0);
    validate_cuda<fp32_t>(35, 0.0, 1.0);
    validate_cuda<fp64_t>(35, 1.0, -1.0);
    validate_cuda<fp64_t>(35, 2.0, 0.0);
    validate_cuda<fp64_t>(35, 0.0, 1.0);
#endif // NNTILE_USE_CUDA

    return 0;
}
