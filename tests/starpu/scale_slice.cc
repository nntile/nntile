/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/starpu/scale_slice.cc
 * Tests for scale_slice operation
 *
 * @version 1.1.0
 * */

#include "nntile/starpu/scale_slice.hh"
#include "nntile/starpu/add_slice_inplace.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::starpu;

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
    // Create handles
    Handle src_handle(src.data(), m * n * sizeof(T));
    Handle dst_handle(dst.data(), m * k * n * sizeof(T));
    Handle dst_ref_handle(dst_ref.data(), m * k * n * sizeof(T));
    // Launch scale_slice
    scale_slice.submit<T>(m, n, k, alpha, src_handle, dst_handle);
    // Launch add_slice_inplace with beta=0 for reference
    add_slice_inplace.submit<T>(m, n, k, alpha, src_handle, T{0}, dst_ref_handle);
    // Wait for completion
    starpu_task_wait_for_all();
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
    // Initialize StarPU
    starpu_init(nullptr);
    // Test different types
    test_scale_slice<fp32_t>();
    test_scale_slice<fp64_t>();
    // Shutdown StarPU
    starpu_shutdown();
    return 0;
}
