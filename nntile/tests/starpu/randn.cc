/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/starpu/randn.cc
 * Smart randn StarPU buffer
 *
 * @version 1.1.0
 * */

#include "nntile/context.hh"
#include "nntile/starpu/randn.hh"
#include "nntile/kernel/randn.hh"
#include "../testing.hh"
#include <array>
#include <vector>
#include <stdexcept>
#include <iostream>
#ifdef NNTILE_USE_CUDA
#include <cuda_runtime.h>
#endif

using namespace nntile;
using namespace nntile::starpu;

template<typename T, std::size_t NDIM>
void validate_cpu(std::array<Index, NDIM> start, std::array<Index, NDIM> shape,
        std::array<Index, NDIM> underlying_shape)
{
    using Y = typename T::repr_t;
    // Randn related constants
    Scalar mean = 2, stddev = 4;
    unsigned long long seed = -1;
    // Strides and number of elements
    Index nelems = shape[0];
    std::vector<Index> stride(NDIM);
    stride[0] = 2; // Custom stride
    Index size = stride[0]*(shape[0]-1) + 1;
    for(Index i = 1; i < NDIM; ++i)
    {
        stride[i] = stride[i-1]*shape[i-1] + 1; // Custom stride
        size += stride[i] * (shape[i]-1);
        nelems *= shape[i];
    }
    // Init all the data
    std::vector<T> data(size);
    for(Index i = 0; i < size; ++i)
    {
        data[i] = Y(i+1);
    }
    // Create copies of data
    std::vector<T> data2(data);
    // Launch low-level kernel
    std::vector<nntile::int64_t> tmp_index(NDIM);
    std::cout << "Run kernel::randn::cpu<" << T::short_name << ">\n";
    kernel::randn::cpu<T>(NDIM, nelems, seed, mean, stddev, &start[0],
            &shape[0], &underlying_shape[0], &data[0], &stride[0],
            &tmp_index[0]);
    // Check by actually submitting a task
    VariableHandle data2_handle(&data2[0], sizeof(T)*size),
        tmp_handle(&tmp_index[0], sizeof(Index)*NDIM);
    std::vector<Index> start_(start.cbegin(), start.cend()),
        shape_(shape.cbegin(), shape.cend()),
        underlying_shape_(underlying_shape.cbegin(), underlying_shape.cend());
    randn.restrict_where(STARPU_CPU);
    std::cout << "Run starpu::randn::submit<" << T::short_name << "> restricted to CPU\n";
    randn.submit<std::tuple<T>>(-1, NDIM, nelems, seed, mean, stddev, start_, shape_,
            stride, underlying_shape_, data2_handle, tmp_handle);
    starpu_task_wait_for_all();
    data2_handle.unregister();
    // Check result
    for(Index i = 0; i < size; ++i)
    {
        TEST_ASSERT(Y(data[i]) == Y(data2[i]));
    }
    std::cout << "OK: starpu::randn::submit<" << T::short_name << "> restricted to CPU\n";
}

#ifdef NNTILE_USE_CUDA
template<typename T, std::size_t NDIM>
void validate_cuda(std::array<Index, NDIM> start, std::array<Index, NDIM> shape,
        std::array<Index, NDIM> underlying_shape)
{
    using Y = typename T::repr_t;
    int cuda_worker_id = starpu_worker_get_by_type(STARPU_CUDA_WORKER, 0);
    if(cuda_worker_id < 0)
    {
        std::cout << "SKIP: no CUDA worker for randn\n";
        return;
    }
    int dev_id = starpu_worker_get_devid(cuda_worker_id);
    cudaError_t cuda_err = cudaSetDevice(dev_id);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cudaStream_t stream;
    cuda_err = cudaStreamCreate(&stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    Scalar mean = 2, stddev = 4;
    unsigned long long seed = -1;
    Index nelems = shape[0];
    std::vector<Index> stride(NDIM);
    stride[0] = 2;
    Index size = stride[0]*(shape[0]-1) + 1;
    for(Index i = 1; i < NDIM; ++i)
    {
        stride[i] = stride[i-1]*shape[i-1] + 1;
        size += stride[i] * (shape[i]-1);
        nelems *= shape[i];
    }
    std::vector<T> data(size);
    for(Index i = 0; i < size; ++i)
    {
        data[i] = Y(i+1);
    }
    std::vector<T> data2(data);
    std::vector<nntile::int64_t> tmp_index(NDIM);
    kernel::randn::cpu<T>(NDIM, nelems, seed, mean, stddev, &start[0],
            &shape[0], &underlying_shape[0], &data[0], &stride[0],
            &tmp_index[0]);
    T *dev_data;
    cuda_err = cudaMalloc(&dev_data, sizeof(T)*size);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_data, &data2[0], sizeof(T)*size,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    std::cout << "Run kernel::randn::cuda<" << T::short_name << ">\n";
    kernel::randn::cuda<T>(stream, NDIM, nelems, seed, mean, stddev, &start[0],
            &shape[0], &underlying_shape[0], dev_data, &stride[0]);
    cuda_err = cudaStreamSynchronize(stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaStreamDestroy(stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    std::vector<T> data_cuda(size);
    cuda_err = cudaMemcpy(&data_cuda[0], dev_data, sizeof(T)*size,
            cudaMemcpyDeviceToHost);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaFree(dev_data);
    TEST_ASSERT(cuda_err == cudaSuccess);
    for(Index i = 0; i < size; ++i)
    {
        TEST_ASSERT(Y(data[i]) == Y(data_cuda[i]));
    }
    VariableHandle data2_handle(&data2[0], sizeof(T)*size),
        tmp_handle(&tmp_index[0], sizeof(Index)*NDIM);
    std::vector<Index> start_(start.cbegin(), start.cend()),
        shape_(shape.cbegin(), shape.cend()),
        underlying_shape_(underlying_shape.cbegin(), underlying_shape.cend());
    randn.restrict_where(STARPU_CUDA);
    std::cout << "Run starpu::randn::submit<" << T::short_name << "> restricted to CUDA\n";
    randn.submit<std::tuple<T>>(-1, NDIM, nelems, seed, mean, stddev, start_, shape_,
            stride, underlying_shape_, data2_handle, tmp_handle);
    starpu_task_wait_for_all();
    data2_handle.unregister();
    for(Index i = 0; i < size; ++i)
    {
        TEST_ASSERT(Y(data[i]) == Y(data2[i]));
    }
    std::cout << "OK: starpu::randn::submit<" << T::short_name << "> restricted to CUDA\n";
}
#endif // NNTILE_USE_CUDA

// Run multiple tests for a given precision
template<typename T>
void validate_many()
{
    validate_cpu<T, 1>({0}, {1}, {2});
    validate_cpu<T, 1>({2}, {1}, {4});
    validate_cpu<T, 1>({0}, {2}, {2});
    validate_cpu<T, 3>({0, 0, 0}, {1, 2, 4}, {2, 3, 4});
    validate_cpu<T, 3>({1, 0, 0}, {1, 3, 4}, {2, 3, 4});
    validate_cpu<T, 3>({1, 0, 0}, {1, 2, 2}, {2, 3, 4});
    validate_cpu<T, 3>({0, 1, 2}, {2, 2, 2}, {2, 3, 4});
#ifdef NNTILE_USE_CUDA
    validate_cuda<T, 1>({0}, {1}, {2});
    validate_cuda<T, 1>({2}, {1}, {4});
    validate_cuda<T, 1>({0}, {2}, {2});
    validate_cuda<T, 3>({0, 0, 0}, {1, 2, 4}, {2, 3, 4});
    validate_cuda<T, 3>({1, 0, 0}, {1, 3, 4}, {2, 3, 4});
    validate_cuda<T, 3>({1, 0, 0}, {1, 2, 2}, {2, 3, 4});
    validate_cuda<T, 3>({0, 1, 2}, {2, 2, 2}, {2, 3, 4});
#endif
}

int main(int argc, char **argv)
{
    // Initialize StarPU (it will automatically shutdown itself on exit)
#ifdef NNTILE_USE_CUDA
    int ncpu=1, ncuda=1, ooc=0, verbose=0;
#else
    int ncpu=1, ncuda=0, ooc=0, verbose=0;
#endif
    const char *ooc_path = "/tmp/nntile_ooc";
    size_t ooc_size = 16777216;
    auto context = Context(ncpu, ncuda, ooc, ooc_path, ooc_size, verbose);

    // Launch all tests
    validate_many<fp32_t>();
    validate_many<fp64_t>();

    return 0;
}
