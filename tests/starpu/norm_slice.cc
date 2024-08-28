/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/starpu/norm_slice.cc
 * Euclidean norms of fibers into a slice of a StarPU buffer
 *
 * @version 1.1.0
 * */

#include "nntile/starpu/norm_slice.hh"
#include "nntile/kernel/norm_slice.hh"
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
void validate_cpu(Index m, Index n, Index k, Scalar alpha, Scalar beta)
{
    using Y = typename T::repr_t;
    // Init all the data
    std::vector<T> src(m*n*k);
    for(Index i = 0; i < m*n*k; ++i)
    {
        src[i] = Y(i+1);
    }
    std::vector<T> dst(m*n);
    for(Index i = 0; i < m*n; i += 1)
    {
        dst[i] = Y(i+0.5);
    }
    // Create copies of destination
    std::vector<T> dst2(dst);
    // Launch low-level kernel
    std::cout << "Run kernel::norm_slice::cpu<" << T::type_repr << ">\n";
    kernel::norm_slice::cpu<T>(m, n, k, alpha, &src[0], beta, &dst[0]);
    // Check by actually submitting a task
    VariableHandle src_handle(&src[0], sizeof(T)*m*n*k, STARPU_R),
        dst2_handle(&dst2[0], sizeof(T)*m*n, STARPU_RW);
    norm_slice::restrict_where(STARPU_CPU);
    std::cout << "Run starpu::norm_slice::submit<" << T::type_repr << "> restricted to CPU\n";
    norm_slice::submit<T>(m, n, k, alpha, src_handle, beta, dst2_handle);
    starpu_task_wait_for_all();
    dst2_handle.unregister();
    // Check result
    for(Index i = 0; i < m*n; ++i)
    {
        TEST_ASSERT(Y(dst[i]) == Y(dst2[i]));
    }
    std::cout << "OK: starpu::norm_slice::submit<" << T::type_repr << "> restricted to CPU\n";
}

int main(int argc, char **argv)
{
    // Init StarPU for testing
    Config starpu(1, 1, 0);
    // Init codelet
    norm_slice::init();
    // Launch all tests
    validate_cpu<fp32_t>(3, 5, 7, 1.0, -1.0);
    validate_cpu<fp32_t>(3, 5, 7, 2.0, 0.0);
    validate_cpu<fp32_t>(3, 5, 7, 0.0, 1.0);
    validate_cpu<fp64_t>(3, 5, 7, 1.0, -1.0);
    validate_cpu<fp64_t>(3, 5, 7, 2.0, 0.0);
    validate_cpu<fp64_t>(3, 5, 7, 0.0, 1.0);
    return 0;
}
