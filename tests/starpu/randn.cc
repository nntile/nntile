/*! @randnright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/starpu/randn.cc
 * Smart randn StarPU buffer
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-12
 * */

#include "nntile/starpu/randn.hh"
#include "nntile/kernel/cpu/randn.hh"
#include <array>
#include <vector>
#include <stdexcept>

using namespace nntile;

template<typename T, std::size_t NDIM>
void validate_cpu(std::array<Index, NDIM> start, std::array<Index, NDIM> shape,
        std::array<Index, NDIM> underlying_shape)
{
    // Randn related constants
    T mean = 2, stddev = 4;
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
        data[i] = T(i+1);
    }
    // Create copies of data
    std::vector<T> data2(data), data3(data);
    // Launch low-level kernel
    std::vector<Index> tmp_index(NDIM);
    kernel::cpu::randn<T>(NDIM, nelems, seed, mean, stddev, &start[0],
            &shape[0], &underlying_shape[0], &data[0], &stride[0],
            &tmp_index[0]);
    // Launch corresponding StarPU codelet
    void *args;
    std::size_t args_size;
    Index ndim = NDIM;
    starpu_codelet_pack_args(&args, &args_size,
            STARPU_VALUE, &ndim, sizeof(ndim),
            STARPU_VALUE, &nelems, sizeof(nelems),
            STARPU_VALUE, &seed, sizeof(seed),
            STARPU_VALUE, &mean, sizeof(mean),
            STARPU_VALUE, &stddev, sizeof(stddev),
            STARPU_VALUE, &start[0], NDIM*sizeof(start[0]),
            STARPU_VALUE, &shape[0], NDIM*sizeof(shape[0]),
            STARPU_VALUE, &stride[0], NDIM*sizeof(stride[0]),
            STARPU_VALUE, &underlying_shape[0],
            NDIM*sizeof(underlying_shape[0]),
            0);
    StarpuVariableInterface
        data2_interface(&data2[0], sizeof(T)*size),
        tmp_interface(&tmp_index[0], sizeof(Index)*NDIM);
    void *buffers[2] = {&data2_interface, &tmp_interface};
    starpu::randn_cpu<T>(buffers, args);
    free(args);
    // Check result
    for(Index i = 0; i < size; ++i)
    {
        if(data[i] != data2[i])
        {
            throw std::runtime_error("StarPU codelet wrong result");
        }
    }
    // Check by actually submitting a task
    StarpuVariableHandle data3_handle(&data3[0], sizeof(T)*size),
        tmp_handle(&tmp_index[0], sizeof(Index)*NDIM);
    starpu::randn_restrict_where(STARPU_CPU);
    std::vector<Index> start_(start.cbegin(), start.cend()),
        shape_(shape.cbegin(), shape.cend()),
        underlying_shape_(underlying_shape.cbegin(), underlying_shape.cend());
    starpu_resume();
    starpu::randn<T>(NDIM, nelems, seed, mean, stddev, start_, shape_, stride,
            underlying_shape_, data3_handle, tmp_handle);
    starpu_task_wait_for_all();
    data3_handle.unregister();
    starpu_pause();
    // Check result
    for(Index i = 0; i < size; ++i)
    {
        if(data[i] != data3[i])
        {
            throw std::runtime_error("StarPU submission wrong result");
        }
    }
}

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
}

int main(int argc, char **argv)
{
    // Init StarPU configuration and set number of CPU workers to 1
    starpu_conf conf;
    int ret = starpu_conf_init(&conf);
    if(ret != 0)
    {
        throw std::runtime_error("starpu_conf_init error");
    }
    conf.ncpus = 1;
    conf.ncuda = 0;
    ret = starpu_init(&conf);
    if(ret != 0)
    {
        throw std::runtime_error("starpu_init error");
    }
    // Launch all tests
    starpu_pause();
    validate_many<fp32_t>();
    validate_many<fp64_t>();
    starpu_resume();
    starpu_shutdown();
    return 0;
}

