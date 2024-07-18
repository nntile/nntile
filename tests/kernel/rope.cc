/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/rope.cc
 * RoPE operation on a buffer
 *
 * @version 1.0.0
 * */

#include "nntile/kernel/rope.hh"
#include "../testing.hh"
#include <vector>
#include <stdexcept>
#include <cmath>
#include <iostream>

using namespace nntile;
using namespace nntile::kernel::rope;

#ifdef NNTILE_USE_CUDA
// Check low-level CUDA kernel
#endif // NNTILE_USE_CUDA

// Templated validation
template<typename T>
void validate(Index m, Index n)
{
    using Y = typename T::repr_t;
    const Y eps = 2 * T::epsilon();
    Index num_data_elems{2*m*n};
    // Init test input
    std::vector<T> sin(m);
    std::vector<T> cos(m);
    std::vector<T> src(num_data_elems);
    std::vector<T> dst(num_data_elems);
    for(Index i = 0; i < num_data_elems; ++i)
    {
        src[i] = Y(2*i+1-num_data_elems) / Y{1000};
    }

    for(Index i = 0; i < m; ++i)
    {
        sin[i] = Y(2*i+1-m) / Y(m);
        cos[i] = Y(std::sqrt(1 - Y(sin[i]) * Y(sin[i])));
    }

    // Check low-level CPU kernel
    std::cout << "Run kernel::rope::cpu<" << T::type_repr << ">\n";
    cpu<T>(m, n, &sin[0], &cos[0], &src[0], &dst[0]);
    for(Index j = 0; j < n; ++j)
    {
        for(Index i = 0; i < m; ++i)
        {
            Index l = 2 * (i+j*m);
            Y c{cos[i]}, s{sin[i]};
            Y a{src[l]}, b{src[l+1]};
            Y val_ref_a{c*a - s*b};
            Y val_ref_b{s*a + c*b};

            // Obtain range of correct values
            Y val_ref_a_min, val_ref_a_max;
            Y val_ref_b_min, val_ref_b_max;
            if(val_ref_a < 0)
            {
                val_ref_a_min = val_ref_a * (Y{1}+eps) - eps;
                val_ref_a_max = val_ref_a * (Y{1}-eps) + eps;
            }
            else
            {
                val_ref_a_min = val_ref_a * (Y{1}-eps) - eps;
                val_ref_a_max = val_ref_a * (Y{1}+eps) + eps;
            }

            if(val_ref_b < 0)
            {
                val_ref_b_min = val_ref_b * (Y{1}+eps) - eps;
                val_ref_b_max = val_ref_b * (Y{1}-eps) + eps;
            }
            else
            {
                val_ref_b_min = val_ref_b * (Y{1}-eps) - eps;
                val_ref_b_max = val_ref_b * (Y{1}+eps) + eps;
            }

            // NaN-aware comparisons
            TEST_ASSERT(Y(dst[l]) >= val_ref_a_min and Y(dst[l]) <= val_ref_a_max);
            TEST_ASSERT(Y(dst[l+1]) >= val_ref_b_min and Y(dst[l+1]) <= val_ref_b_max);
        }
    }
    std::cout << "OK: kernel::rope::cpu<" << T::type_repr << ">\n";
#ifdef NNTILE_USE_CUDA
    // Check low-level CUDA kernel
#endif // NNTILE_USE_CUDA
}

int main(int argc, char **argv)
{
    validate<fp32_t>(0,5);
    validate<fp32_t>(1,2);
    validate<fp32_t>(1000,100);
    validate<fp64_t>(0,5);
    validate<fp64_t>(1,2);
    validate<fp64_t>(1000,100);
    return 0;
}
