/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/starpu/gelu.cc
 * GeLU operation on a StarPU buffer
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-12
 * */

#include "nntile/starpu/gelu.hh"
#include "nntile/kernel/cpu/gelu.hh"
#include <vector>
#include <stdexcept>

using namespace nntile;

template<typename T>
void validate_cpu(Index nelems)
{
    // Init all the data
    std::vector<T> data(nelems);
    for(Index i = 0; i < nelems; ++i)
    {
        data[i] = T(i+1);
    }
    // Create copies of destination
    std::vector<T> data2(data), data3(data);
    // Launch low-level kernel
    kernel::cpu::gelu<T>(nelems, &data[0]);
    // Launch corresponding StarPU codelet
    StarpuVariableInterface data2_interface(&data2[0], sizeof(T)*nelems);
    void *buffers[1] = {&data2_interface};
    starpu::gelu_cpu<T>(buffers, &nelems);
    // Check result
    for(Index i = 0; i < nelems; ++i)
    {
        if(data[i] != data2[i])
        {
            throw std::runtime_error("StarPU codelet wrong result");
        }
    }
    // Check by actually submitting a task
    StarpuVariableHandle data3_handle(&data3[0], sizeof(T)*nelems);
    starpu::gelu_restrict_where(STARPU_CPU);
    starpu_resume();
    starpu::gelu<T>(nelems, data3_handle);
    starpu_task_wait_for_all();
    data3_handle.unregister();
    starpu_pause();
    // Check result
    for(Index i = 0; i < nelems; ++i)
    {
        if(data[i] != data3[i])
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
    validate_cpu<fp32_t>(0);
    validate_cpu<fp32_t>(1);
    validate_cpu<fp32_t>(10000);
    validate_cpu<fp64_t>(0);
    validate_cpu<fp64_t>(1);
    validate_cpu<fp64_t>(10000);
    starpu_resume();
    starpu_shutdown();
    return 0;
}

