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
 * @date 2022-08-12
 * */

#include "nntile/starpu/clear.hh"
#include <vector>
#include <stdexcept>

using namespace nntile;

void validate_cpu(std::size_t size)
{
    // Init all the data
    std::vector<char> data(size);
    for(std::size_t i = 0; i < size; ++i)
    {
        data[i] = -1;
    }
    std::vector<char> data2(data);
    // Launch StarPU codelet
    StarpuVariableInterface data_interface(&data[0], size);
    void *buffers[1] = {&data_interface};
    starpu::clear_cpu(buffers, nullptr);
    // Check result
    for(std::size_t i = 0; i < size; ++i)
    {
        if(data[i] != 0)
        {
            throw std::runtime_error("StarPU codelet wrong result");
        }
    }
    // Check by actually submitting a task
    StarpuVariableHandle data2_handle(&data2[0], size);
    starpu::clear_restrict_where(STARPU_CPU);
    starpu_resume();
    starpu::clear(data2_handle);
    starpu_task_wait_for_all();
    data2_handle.unregister();
    starpu_pause();
    // Check result
    for(std::size_t i = 0; i < size; ++i)
    {
        if(data2[i] != 0)
        {
            throw std::runtime_error("StarPU submission wrong result");
        }
    }
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
    validate_cpu(0);
    validate_cpu(1);
    validate_cpu(100000);
    starpu_resume();
    starpu_shutdown();
    return 0;
}

