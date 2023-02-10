/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/starpu/add_scalar.cc
 * Add scalar to StarPU buffers
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @date 2023-02-10
 * */

#include "nntile/starpu/add_scalar.hh"
#include "../testing.hh"
#include <vector>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <limits>

using namespace nntile;
using namespace nntile::starpu;

template<typename T>
void validate_cpu(T val, Index num_elements)
{
    constexpr T eps = std::numeric_limits<T>::epsilon();
    // Init all the data
    std::vector<T> src(num_elements);
    for (Index i = 0; i < num_elements; ++i)
        src[i] = T(i + 1);
    std::vector<T> src_copy(src);
    // Check by actually submitting a task
    VariableHandle src_handle(&src[0], sizeof(T) * num_elements, STARPU_RW);
    add_scalar::restrict_where(STARPU_CPU);
    std::cout << "Run starpu::add_scalar::submit<T> restricted to CPU\n";
    add_scalar::submit<T>(val, num_elements, src_handle);
    starpu_task_wait_for_all();
    src_handle.unregister();
    // Check result
    for (Index i = 0; i < num_elements; ++i)
        TEST_ASSERT(src_copy[i] + val == src[i]);
    std::cout << "OK: starpu::add_scalar::submit<T> restricted to CPU\n";
}

int main(int argc, char **argv)
{
    // Init StarPU for testing
    Config starpu(1, 0, 0);
    // Init codelet
    add_scalar::init();
    // Launch all tests
    validate_cpu<fp32_t>(10, 1000);
    validate_cpu<fp32_t>(-10.34, 10);
    validate_cpu<fp64_t>(10, 1000);
    validate_cpu<fp64_t>(-10.34, 10);
    return 0;
}
