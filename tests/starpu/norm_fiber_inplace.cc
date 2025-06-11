/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/starpu/norm_fiber_inplace.cc
 * Euclidean norms over slices into a fiber of a product of a StarPU buffer
 *
 * @version 1.1.0
 * */

#include "nntile/context.hh"
#include "nntile/starpu/norm_fiber_inplace.hh"
#include "nntile/kernel/norm_fiber_inplace.hh"
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
void validate_cpu(Index m, Index n, Index k, Index batch, Scalar alpha, Scalar beta)
{
    using Y = typename T::repr_t;
    const Y eps = T::epsilon;
    // Init all the data
    std::vector<T> src1(m*n*k*batch);
    std::vector<T> dst(k*batch);
    T *src_pointer = &src1[0];
    for(Index b = 0; b < batch; ++b) {
        for(Index i2 = 0; i2 < k; ++i2)
        {
            dst[b*batch+i2] = Y{0.0};
            for(Index i1 = 0; i1 < n; ++i1)
            {
                T *src_slice = src_pointer + ((i1+b*n)*k+i2)*m;
                for(Index i0 = 0; i0 < m; ++i0)
                {
                    src_slice[i0] = Y{-1.0};
                }
            }
        }
    }
    // Create copies of destination
    std::vector<T> dst2(dst);
    // Launch low-level kernel
    std::cout << "Run kernel::norm_fiber_inplace::cpu<" << T::short_name << ">\n";
    kernel::norm_fiber_inplace::cpu<T>(m, n, k, batch, alpha, &src1[0], beta, &dst[0]);
    // Check by actually submitting a task
    VariableHandle src1_handle(&src1[0], sizeof(T)*m*n*k*batch);
    VariableHandle dst2_handle(&dst2[0], sizeof(T)*k*batch);
    norm_fiber_inplace.restrict_where(STARPU_CPU);
    std::cout << "Run starpu::norm_fiber_inplace::submit<" << T::short_name << "> restricted to CPU\n";
    int redux = 0;
    norm_fiber_inplace.submit<std::tuple<T>>(m, n, k, batch, alpha, src1_handle, beta, dst2_handle, redux);
    starpu_task_wait_for_all();
    dst2_handle.unregister();
    // Check result
    for(Index i = 0; i < dst.size(); ++i)
    {
        TEST_ASSERT(Y(dst[i]) == Y(dst2[i]));
    }
    std::cout << "OK: starpu::norm_fiber_inplace::submit<" << T::short_name << "> restricted to CPU\n";
}

int main(int argc, char **argv)
{
    // Initialize StarPU
    int ncpu=1, ncuda=1, ooc=0, verbose=0;
    const char *ooc_path = "/tmp/nntile_ooc";
    size_t ooc_size = 16777216;
    auto context = Context(ncpu, ncuda, ooc, ooc_path, ooc_size, verbose);

    // Launch all tests
    validate_cpu<fp64_t>(32, 32, 10, 1, 1.0, 0.0);
    validate_cpu<fp64_t>(32, 9, 10, 1, 1.0, 0.0);
    validate_cpu<fp32_t>(32, 32, 10, 1, 1.0, 0.0);
    validate_cpu<fp32_t>(32, 9, 10, 1, 1.0, 0.0);
    return 0;
}
