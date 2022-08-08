/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/cpu/bias.cc
 * Bias operation on a buffer on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-04
 * */

#include "nntile/kernel/cpu/bias.hh"
#include <vector>
#include <stdexcept>
#include <limits>

using namespace nntile;
using namespace nntile::kernel::cpu;

// Templated validation
template<typename T>
void validate(Index m, Index n, Index k)
{
    // Init test input
    std::vector<T> src(m*n), dst(m*n*k);
    for(Index i0 = 0; i0 < m; ++i0)
    {
        for(Index i1 = 0; i1 < n; ++i1)
        {
            for(Index i2 = 0; i2 < k; ++i2)
            {
                dst[(i1*k+i2)*m+i0] = T(i0+i1+i2) / T{10};
            }
        }
    }
    for(Index i0 = 0; i0 < m; ++i0)
    {
        for(Index i1 = 0; i1 < n; ++i1)
        {
            src[i1*m+i0] = T(-i0-i1) / T{10};
        }
    }
    std::vector<T> dst2(dst);
    // Check low-level kernel
    bias<T>(m, n, k, &src[0], &dst[0]);
    for(Index i0 = 0; i0 < m; ++i0)
    {
        for(Index i1 = 0; i1 < n; ++i1)
        {
            for(Index i2 = 0; i2 < k; ++i2)
            {
                T val = dst[(i1*k+i2)*m+i0];
                T val_ref = T(i2) / T{10};
                if(std::abs(val/val_ref-T{1})
                        / std::numeric_limits<T>::epsilon() > 10)
                {
                    throw std::runtime_error("Wrong value");
                }
            }
        }
    }
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

