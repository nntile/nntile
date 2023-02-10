/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/add_scalar.cc
 * Add scalar to elements from buffer
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @date 2023-02-10
 * */

#include "nntile/kernel/add_scalar.hh"
#include "nntile/base_types.hh"
#include "../testing.hh"
#include <vector>
#include <stdexcept>
#include <limits>
#include <cmath>
#include <iostream>

using namespace nntile;
using namespace nntile::kernel::add_scalar;

// Templated validation
template<typename T>
void validate(T val, Index num_elements)
{
    constexpr T eps = std::numeric_limits<T>::epsilon();
    // Init test input
    std::vector<T> x(num_elements);
    for (Index i = 0; i < num_elements; ++i)
        x[i] = i / rand();
    std::vector<T> y(x);
    // Check low-level kernel
    std::cout << "Run kernel::add_scalar::cpu<T>\n";
    cpu<T>(val, num_elements, &x[0]);
    for (Index i = 0; i < num_elements; ++i)
        TEST_ASSERT(y[i] + val == x[i]);
    std::cout << "OK: kernel::add_scalar::cpu<T>\n";
}

int main(int argc, char **argv)
{
    validate<fp32_t>(10, 100);
    validate<fp64_t>(10.5, 1000);
    return 0;
}
