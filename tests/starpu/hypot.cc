/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/starpu/hypot.cc
 * Hypot operation for StarPU buffers
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-18
 * */

#include "nntile/starpu/hypot.hh"
#include "../testing.hh"
#include <vector>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <limits>

using namespace nntile;
using namespace nntile::starpu;

template<typename T>
void validate_cpu()
{
    constexpr T eps = std::numeric_limits<T>::epsilon();
    // Init all the data
    std::vector<T> src(1), dst(1);
    src[0] = T{1};
    dst[0] = T{2};
    T alpha = 0.5, beta = -1.5;
    // Check by actually submitting a task
    VariableHandle src_handle(&src[0], sizeof(T), STARPU_R),
        dst_handle(&dst[0], sizeof(T), STARPU_RW);
    hypot::restrict_where(STARPU_CPU);
    std::cout << "Run starpu::hypot::submit<T> restricted to CPU\n";
    hypot::submit<T>(alpha, src_handle, beta, dst_handle);
    starpu_task_wait_for_all();
    dst_handle.unregister();
    // Check result
    TEST_ASSERT(std::abs(dst[0]*dst[0]-T{9.25}) <= 10*eps);
    std::cout << "OK: starpu::hypot::submit<T> restricted to CPU\n";
}

int main(int argc, char **argv)
{
    // Init StarPU for testing
    Config starpu(1, 0, 0);
    // Init codelet
    hypot::init();
    // Launch all tests
    validate_cpu<fp32_t>();
    validate_cpu<fp64_t>();
    return 0;
}

