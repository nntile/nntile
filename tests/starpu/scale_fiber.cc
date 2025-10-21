/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/starpu/scale_fiber.cc
 * StarPU wrappers for scaling of a tensor with a broadcasted fiber
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/starpu/scale_fiber.hh"

// Standard libraries
#include <vector>
#include <stdexcept>
#include <limits>
#include <iostream>
#include <cmath>
#include <random>
#include <string>

// Third-party libraries
#include <catch2/catch_all.hpp>

// Other NNTile headers
#include <nntile/kernel/scale_fiber.hh>

// Use namespaces for shorter code
using namespace Catch;
using namespace Catch::Matchers;

// Use tested NNTile namespaces
using namespace nntile;
using namespace nntile::starpu;

// Type to acquire reference values
using ref_t = double;

// Struct to hold test data and reference results
template<typename T>
struct TestData
{
    using Y = typename T::repr_t;
    // Input parameters
    Index m, n, k, batch;
    Scalar alpha;
    // Input data
    std::vector<T> src, dst, dst_ref;
    // Reference results
    std::vector<Y> src_ref, dst_ref_ref;
    // Constructor
    TestData(Index m_, Index n_, Index k_, Index batch_, Scalar alpha_):
        m(m_), n(n_), k(k_), batch(batch_), alpha(alpha_),
        src(k * batch),
        dst(m * k * n * batch),
        dst_ref(m * k * n * batch),
        src_ref(k * batch),
        dst_ref_ref(m * k * n * batch)
    {
        // Fill input data with random values
        std::random_device dev;
        std::mt19937_64 rng(dev());
        std::uniform_real_distribution<Y> dist(-1.0, 1.0);
        for(Index i = 0; i < k * batch; ++i)
        {
            src[i] = T{dist(rng)};
            src_ref[i] = Y{src[i]};
        }
        // Fill output data with random values
        for(Index i = 0; i < m * k * n * batch; ++i)
        {
            dst[i] = T{dist(rng)};
            dst_ref[i] = T{dist(rng)};
        }
        // Compute reference results
        for(Index b = 0; b < batch; ++b)
        {
            for(Index i2 = 0; i2 < k; ++i2)
            {
                const Y src_val = alpha * src_ref[i2 + b * k];
                for(Index i1 = 0; i1 < n; ++i1)
                {
                    for(Index i0 = 0; i0 < m; ++i0)
                    {
                        Index idx = ((i1 + b * n) * k + i2) * m + i0;
                        dst_ref_ref[idx] = src_val;
                    }
                }
            }
        }
    }
};

// Test StarPU wrapper
template<typename T>
void test_starpu(Index m, Index n, Index k, Index batch, Scalar alpha)
{
    // Prepare test data
    TestData<T> data(m, n, k, batch, alpha);
    // Create StarPU handles
    Handle src_handle(k * batch, sizeof(T), STARPU_R),
           dst_handle(m * k * n * batch, sizeof(T), STARPU_W);
    // Copy data to StarPU
    starpu_data_cpy(src_handle.get(), data.src.data(), k * batch, sizeof(T),
            nullptr, nullptr);
    starpu_data_cpy(dst_handle.get(), data.dst.data(), m * k * n * batch, sizeof(T),
            nullptr, nullptr);
    // Submit task
    scale_fiber.submit<std::tuple<T>>(m, n, k, batch, alpha, src_handle, dst_handle);
    // Wait for completion
    starpu_task_wait_for_all();
    // Copy result back
    starpu_data_cpy(data.dst.data(), dst_handle.get(), m * k * n * batch, sizeof(T),
            nullptr, nullptr);
    // Check results
    using Y = typename T::repr_t;
    const Y eps = std::numeric_limits<Y>::epsilon();
    for(Index i = 0; i < m * k * n * batch; ++i)
    {
        Y dst_val = Y{data.dst[i]};
        Y dst_ref_val = data.dst_ref_ref[i];
        REQUIRE(std::abs(dst_val - dst_ref_val) <= eps * std::abs(dst_ref_val));
    }
}

// Test cases
TEST_CASE("scale_fiber StarPU", "[starpu][scale_fiber]")
{
    // Test different sizes and alpha values
    test_starpu<fp32_t>(1, 1, 1, 1, 1.0);
    test_starpu<fp32_t>(2, 3, 4, 5, 2.0);
    test_starpu<fp32_t>(3, 2, 1, 1, 0.5);
    test_starpu<fp32_t>(1, 1, 1, 1, 0.0);
    test_starpu<fp64_t>(1, 1, 1, 1, 1.0);
    test_starpu<fp64_t>(2, 3, 4, 5, 2.0);
    test_starpu<fp64_t>(3, 2, 1, 1, 0.5);
    test_starpu<fp64_t>(1, 1, 1, 1, 0.0);
}
