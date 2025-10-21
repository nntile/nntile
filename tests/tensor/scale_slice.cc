/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tensor/scale_slice.cc
 * Tests for scale_slice operation
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/scale_slice.hh"
#include "nntile/tensor/add_slice_inplace.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tensor;

template<typename T>
void test_scale_slice()
{
    // Test parameters
    constexpr Index m = 3, n = 4, k = 5;
    constexpr T alpha = 2.0;
    // Allocate memory
    std::vector<T> src_data(m * n), dst_data(m * k * n), dst_ref_data(m * k * n);
    // Fill source
    for(Index i = 0; i < m * n; ++i)
    {
        src_data[i] = static_cast<T>(i + 1);
    }
    // Fill destination with zeros
    std::fill(dst_data.begin(), dst_data.end(), T{0});
    std::fill(dst_ref_data.begin(), dst_ref_data.end(), T{0});
    // Create tensors
    Tensor<T> src(src_data.data(), {m, n}, {1, m});
    Tensor<T> dst(dst_data.data(), {m, k, n}, {1, m, m*k});
    Tensor<T> dst_ref(dst_ref_data.data(), {m, k, n}, {1, m, m*k});
    // Launch scale_slice
    scale_slice<T>(alpha, src, dst, 1);
    // Launch add_slice_inplace with beta=0 for reference
    add_slice_inplace<T>(alpha, src, T{0}, dst_ref, 1);
    // Check result
    for(Index i = 0; i < m * k * n; ++i)
    {
        if(dst_data[i] != dst_ref_data[i])
        {
            throw std::runtime_error("Test failed: dst_data[i] != dst_ref_data[i]");
        }
    }
}

int main(int argc, char **argv)
{
    // Initialize StarPU
    starpu_init(nullptr);
    // Test different types
    test_scale_slice<fp32_t>();
    test_scale_slice<fp64_t>();
    // Shutdown StarPU
    starpu_shutdown();
    return 0;
}
