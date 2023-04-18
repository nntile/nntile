/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/starpu/norm.cc
 * Norm for StarPU buffer
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-18
 * */

#include "nntile/starpu/norm.hh"
#include "nntile/kernel/norm.hh"
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
void validate_cpu(Index m, Index n, Index k, T alpha, T beta)
{
    // Init all the data
    std::vector<T> src(m*n*k);
    for(Index i = 0; i < m*n*k; ++i)
    {
        src[i] = T(i+1);
    }
    std::vector<T> norm_dst(m*n);
    for(Index i = 0; i < m*n; i += 1)
    {
        norm_dst[i] = T(i+0.5);
    }
    // Create copies of destination
    std::vector<T> norm_dst2(norm_dst);
    // Launch low-level kernel
    std::cout << "Run kernel::norm::cpu<T>\n";
    kernel::norm::cpu<T>(m, n, k, alpha, &src[0], beta, &norm_dst[0]);
    // Check by actually submitting a task
    VariableHandle src_handle(&src[0], sizeof(T)*m*n*k, STARPU_R),
        norm_dst2_handle(&norm_dst2[0], sizeof(T)*m*n, STARPU_RW);
    norm::restrict_where(STARPU_CPU);
    std::cout << "Run starpu::norm::submit<T> restricted to CPU\n";
    norm::submit<T>(m, n, k, alpha, src_handle, beta, norm_dst2_handle);
    starpu_task_wait_for_all();
    norm_dst2_handle.unregister();
    // Check result
    for(Index i = 0; i < m*n; ++i)
    {
        TEST_ASSERT(norm_dst[i] == norm_dst2[i]);
    }
    std::cout << "OK: starpu::norm::submit<T> restricted to CPU\n";
}

int main(int argc, char **argv)
{
    // Init StarPU for testing
    Config starpu(1, 1, 0);
    // Init codelet
    norm::init();
    // Launch all tests
    validate_cpu<fp32_t>(3, 5, 7, 1.0, -1.0);
    validate_cpu<fp32_t>(3, 5, 7, 2.0, 0.0);
    validate_cpu<fp32_t>(3, 5, 7, 0.0, 1.0);
    validate_cpu<fp64_t>(3, 5, 7, 1.0, -1.0);
    validate_cpu<fp64_t>(3, 5, 7, 2.0, 0.0);
    validate_cpu<fp64_t>(3, 5, 7, 0.0, 1.0);
    return 0;
}

