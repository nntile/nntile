/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/sum.cc
 * Sum of a buffer on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @author Konstantin Sozykin
 * @date 2023-02-27
 * */

#include "nntile/kernel/sum.hh"
#include "../testing.hh"
#include <vector>
#include <stdexcept>
#include <limits>
#include <cmath>
#include <iostream>

using namespace nntile;
using namespace nntile::kernel::sum;


// Templated validation
template<typename T>
void validate(Index m, Index n, Index k)
{
    constexpr T eps = std::numeric_limits<T>::epsilon();
    // Init test input
    std::vector<T> src(m*n*k), sum_dst(m*n);
    for(Index i0 = 0; i0 < m; ++i0)
    {
        for(Index i1 = 0; i1 < n; ++i1)
        {
            for(Index i2 = 0; i2 < k; ++i2)
            {
                src[(i1*k+i2)*m+i0] = T(i0+i1+i2) / T{10};
            }
        }
    }
    std::vector<T> sum_copy(sum_dst);
    // Check low-level kernel
    std::cout << "Run kernel::sum::cpu<T>\n";
    cpu<T>(m, n, k, &src[0], &sum_dst[0]);
    for(Index i0 = 0; i0 < m; ++i0)
    {
        for(Index i1 = 0; i1 < n; ++i1)
        {
            Index a = i0 + i1;
            T sum_ref = k * (2*a+k-1) / 2 / T{10};
            T sum = sum_dst[i1*m+i0];
            if(sum_ref == T{0})
            {
                TEST_ASSERT(std::abs(sum) <= 10*eps);
            }
            else
            {
                TEST_ASSERT(std::abs(sum/sum_ref-T{1}) <= 10*eps);
            }
            
        }
    }
    std::cout << "OK: kernel::sum::cpu<T>\n";

}

int main(int argc, char **argv)
{
    validate<fp32_t>(1, 9, 10);
    validate<fp32_t>(8, 9, 1);
    validate<fp32_t>(8, 1, 10);
    validate<fp32_t>(4, 7, 8);
    validate<fp64_t>(1, 9, 10);
    validate<fp64_t>(8, 9, 1);
    validate<fp64_t>(8, 1, 10);
    validate<fp64_t>(4, 7, 8);
    return 0;
}

