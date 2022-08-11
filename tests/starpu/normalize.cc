/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/starpu/normalize.cc
 * Normalize operation on a StarPU buffer
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-10
 * */

#include "nntile/starpu/normalize.hh"
#include "nntile/kernel/cpu/normalize.hh"
#include <vector>
#include <stdexcept>
#include <cmath>

using namespace nntile;

template<typename T>
void validate_cpu(Index m, Index n, Index k, Index l, T eps, T gamma, T beta)
{
    // Init all the data
    std::vector<T> src(2*m*n);
    for(Index i = 0; i < 2*m*n; i += 2)
    {
        src[i] = T(l*i); // Sum
        src[i+1] = std::sqrt(T(l)*i*i + T(l)); // Norm
    }
    std::vector<T> dst(m*n*k);
    for(Index i = 0; i < m*n*k; ++i)
    {
        dst[i] = T(-i-1);
    }
    // Create copies of destination
    std::vector<T> dst2(dst), dst3(dst);
    // Launch low-level kernel
    kernel::cpu::normalize<T>(m, n, k, l, eps, gamma, beta, &src[0], &dst[0]);
    // Launch corresponding StarPU codelet
    starpu::normalize_args<T> args =
    {
        .m = m,
        .n = n,
        .k = k,
        .l = l,
        .eps = eps
    };
    T gamma_beta[2] = {gamma, beta};
    StarpuVariableInterface src_interface(&src[0], sizeof(T)*2*m*n),
        dst2_interface(&dst2[0], sizeof(T)*m*n*k),
        gamma_beta_interface(gamma_beta, sizeof(gamma_beta));
    void *buffers[3] = {&gamma_beta_interface, &src_interface,
        &dst2_interface};
    starpu::normalize_cpu<T>(buffers, &args);
    // Check result
    for(Index i = 0; i < m*n*k; ++i)
    {
        if(dst[i] != dst2[i])
        {
            throw std::runtime_error("StarPU codelet wrong result");
        }
    }
    // Check by actually submitting a task
    StarpuVariableHandle src_handle(&src[0], sizeof(T)*2*m*n),
        dst3_handle(&dst3[0], sizeof(T)*m*n*k),
        gamma_beta_handle(gamma_beta, sizeof(gamma_beta));
    starpu::normalize_restrict_where(STARPU_CPU);
    starpu_resume();
    starpu::normalize<T>(m, n, k, l, eps, gamma_beta_handle, src_handle,
            dst3_handle);
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

template<typename T>
void validate_many()
{
    validate_cpu<T>(3, 5, 7, 10, 0, 1, 0);
    validate_cpu<T>(3, 5, 7, 2, 10, 2, 1);
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

