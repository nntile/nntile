/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/starpu/clear.cc
 * Clear a StarPU buffer
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-15
 * */

#include "nntile/starpu/clear.hh"
#ifdef NNTILE_USE_CUDA
#   include <cuda_runtime.h>
#endif // NNTILE_USE_CUDA
#include <vector>
#include <stdexcept>
#include <iostream>

using namespace nntile;

void validate(std::size_t size)
{
    // Init all the data
    std::vector<char> data_init(size);
    for(std::size_t i = 0; i < size; ++i)
    {
        data_init[i] = -1;
    }
    // Create copy of data
    std::vector<char> data(data_init);
    // Check by actually submitting a task
    StarpuVariableHandle data_handle(&data[0], size, STARPU_RW);
    starpu::clear_restrict_where(STARPU_CPU);
    starpu_resume();
    std::cout << "Run starpu::clear restricted to CPU\n";
    starpu::clear(data_handle);
    starpu_task_wait_for_all();
    data_handle.unregister();
    starpu_pause();
    // Check result
    for(std::size_t i = 0; i < size; ++i)
    {
        if(data[i] != 0)
        {
            throw std::runtime_error("StarPU submission wrong result");
        }
    }
    std::cout << "OK: starpu::clear restricted to CPU\n";
#ifdef NNTILE_USE_CUDA
    // Check by actually submitting a task
    data = data_init;
    data_handle = StarpuVariableHandle(&data[0], size, STARPU_RW);
    starpu::clear_restrict_where(STARPU_CUDA);
    starpu_resume();
    std::cout << "Run starpu::clear restricted to CUDA\n";
    starpu::clear(data_handle);
    starpu_task_wait_for_all();
    data_handle.unregister();
    starpu_pause();
    // Check result
    for(std::size_t i = 0; i < size; ++i)
    {
        if(data[i] != 0)
        {
            throw std::runtime_error("StarPU submission wrong result");
        }
    }
    std::cout << "OK: starpu::clear restricted to CUDA\n";
#endif // NNTILE_USE_CUDA
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
#ifdef NNTILE_USE_CUDA
    conf.ncuda = 1;
#else // NNTILE_USE_CUDA
    conf.ncuda = 0;
#endif // NNTILE_USE_CUDA
    ret = starpu_init(&conf);
    if(ret != 0)
    {
        throw std::runtime_error("starpu_init error");
    }
    // Launch all tests
    starpu_pause();
    validate(1);
    validate(100000);
    starpu_resume();
    starpu_shutdown();
    return 0;
}

