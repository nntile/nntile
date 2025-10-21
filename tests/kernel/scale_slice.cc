/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/scale_slice.cc
 * Tests for scale_slice operation
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/scale_slice.hh"
#include "nntile/kernel/add_slice_inplace.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::kernel;

template<typename T>
void test_scale_slice()
{
    // Test parameters
    constexpr Index m = 3, n = 4, k = 5;
    constexpr T alpha = 2.0;
    // Allocate memory
    std::vector<T> src(m * n), dst(m * k * n), dst_ref(m * k * n);
    // Fill source
    for(Index i = 0; i < m * n; ++i)
    {
        src[i] = static_cast<T>(i + 1);
    }
    // Fill destination with zeros
    std::fill(dst.begin(), dst.end(), T{0});
    std::fill(dst_ref.begin(), dst_ref.end(), T{0});
    // Launch kernel
    scale_slice::cpu<T>(m, n, k, alpha, src.data(), dst.data());
    // Check result using add_slice_inplace with beta=0
    add_slice_inplace::cpu<T>(m, n, k, alpha, src.data(), T{0}, dst_ref.data());
    // Check result
    for(Index i = 0; i < m * k * n; ++i)
    {
        if(dst[i] != dst_ref[i])
        {
            throw std::runtime_error("Test failed: dst[i] != dst_ref[i]");
        }
    }
}

int main(int argc, char **argv)
{
    // Test different types
    test_scale_slice<fp32_t>();
    test_scale_slice<fp64_t>();
    return 0;
}
