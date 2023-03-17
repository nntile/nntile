/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/starpu/sum.cc
 * Sum and Euclidian norm for StarPU buffer
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @author Konstantin Sozykin 
 * @date 2023-03-14
 * */

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
void validate_cpu(Index m, Index n, Index k)
{
    // Init all the data
    std::vector<T> src(m*n*k);
    for(Index i = 0; i < m*n*k; ++i)
    {
        src[i] = T(i+1);
    }
    std::vector<T> sum_dst(m*n);
    for(Index i = 0; i < m*n; i += 1)
    {
        sum_dst[i] = T(-i-1); // Sum
    }
    // Create copies of destination
    std::vector<T> sum_dst2(sum_dst);
    // Launch low-level kernel
    std::cout << "Run kernel::sum::cpu<T>\n";
    kernel::sum::cpu<T>(m, n, k, &src[0], &sum_dst[0]);
    // Check by actually submitting a task
    VariableHandle src_handle(&src[0], sizeof(T)*m*n*k, STARPU_R),
        sum_dst2_handle(&sum_dst2[0], sizeof(T)*m*n, STARPU_RW);
    sum::restrict_where(STARPU_CPU);
    std::cout << "Run starpu::sum::submit<T> restricted to CPU\n";
    sum::submit<T>(m, n, k, src_handle, sum_dst2_handle);
    starpu_task_wait_for_all();
    sum_dst2_handle.unregister();
    // Check result
    for(Index i = 0; i < m*n; ++i)
    {
        TEST_ASSERT(sum_dst[i] == sum_dst2[i]);
    }
    std::cout << "OK: starpu::sum::submit<T> restricted to CPU\n";
}


int main(int argc, char **argv)
{
    // Init StarPU for testing
    Config starpu(1, 1, 0);
    // Init codelet
    sum::init();
    // Launch all tests
    validate_cpu<fp32_t>(3, 5, 7);
    validate_cpu<fp64_t>(3, 5, 7);

    return 0;
}

