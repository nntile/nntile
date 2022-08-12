/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/starpu/bias.cc
 * Bias operation on a StarPU buffer
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-12
 * */

#include "nntile/starpu/bias.hh"
#include "nntile/kernel/cpu/bias.hh"
#include <vector>
#include <stdexcept>

using namespace nntile;

template<typename T>
void validate_cpu(Index m, Index n, Index k)
{
    // Init all the data
    std::vector<T> src(m*n);
    for(Index i = 0; i < m*n; ++i)
    {
        src[i] = T(i+1);
    }
    std::vector<T> dst(m*n*k);
    for(Index i = 0; i < m*n*k; ++i)
    {
        dst[i] = T(-i-1);
    }
    // Create copies of destination
    std::vector<T> dst2(dst), dst3(dst);
    // Launch low-level kernel
    kernel::cpu::bias<T>(m, n, k, &src[0], &dst[0]);
    // Launch corresponding StarPU codelet
    starpu::bias_args args =
    {
        .m = m,
        .n = n,
        .k = k
    };
    StarpuVariableInterface src_interface(&src[0], sizeof(T)*m*n),
        dst2_interface(&dst2[0], sizeof(T)*m*n*k);
    void *buffers[2] = {&src_interface, &dst2_interface};
    starpu::bias_cpu<T>(buffers, &args);
    // Check result
    for(Index i = 0; i < m*n*k; ++i)
    {
        if(dst[i] != dst2[i])
        {
            throw std::runtime_error("StarPU codelet wrong result");
        }
    }
    // Check by actually submitting a task
    StarpuVariableHandle src_handle(&src[0], sizeof(T)*m*n),
        dst3_handle(&dst3[0], sizeof(T)*m*n*k);
    starpu::bias_restrict_where(STARPU_CPU);
    starpu_resume();
    starpu::bias<T>(m, n, k, src_handle, dst3_handle);
    starpu_task_wait_for_all();
    dst3_handle.unregister();
    starpu_pause();
    // Check result
    for(Index i = 0; i < m*n*k; ++i)
    {
        if(dst[i] != dst3[i])
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
    validate_cpu<fp32_t>(3, 5, 7);
    validate_cpu<fp64_t>(3, 5, 7);
    starpu_resume();
    starpu_shutdown();
    return 0;
}

