/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/scale_fiber.cc
 * Tile wrappers for scaling of a tensor with a broadcasted fiber
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/tile/scale_fiber.hh"

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
#include <nntile/tile/tile.hh>
#include <nntile/starpu/scale_fiber.hh>

// Use namespaces for shorter code
using namespace Catch;
using namespace Catch::Matchers;

// Use tested NNTile namespaces
using namespace nntile;
using namespace nntile::tile;

// Type to acquire reference values
using ref_t = double;

// Test tile wrapper
template<typename T>
void test_tile(Index m, Index n, Index k, Index batch, Scalar alpha)
{
    using Y = typename T::repr_t;
    // Create tiles
    std::vector<Index> src_shape = {k};
    for(Index i = 0; i < batch; ++i)
    {
        src_shape.push_back(1);
    }
    std::vector<Index> dst_shape = {m};
    for(Index i = 0; i < k; ++i)
    {
        dst_shape.push_back(1);
    }
    for(Index i = 0; i < n; ++i)
    {
        dst_shape.push_back(1);
    }
    for(Index i = 0; i < batch; ++i)
    {
        dst_shape.push_back(1);
    }
    Tile<T> src_tile(src_shape);
    Tile<T> dst_tile(dst_shape);
    // Fill input data with random values
    std::random_device dev;
    std::mt19937_64 rng(dev());
    std::uniform_real_distribution<Y> dist(-1.0, 1.0);
    for(Index i = 0; i < src_tile.nelems; ++i)
    {
        src_tile[i] = T{dist(rng)};
    }
    // Fill output data with random values
    for(Index i = 0; i < dst_tile.nelems; ++i)
    {
        dst_tile[i] = T{dist(rng)};
    }
    // Compute reference results
    std::vector<Y> dst_ref(dst_tile.nelems);
    for(Index b = 0; b < batch; ++b)
    {
        for(Index i2 = 0; i2 < k; ++i2)
        {
            const Y src_val = alpha * Y{src_tile[i2 + b * k]};
            for(Index i1 = 0; i1 < n; ++i1)
            {
                for(Index i0 = 0; i0 < m; ++i0)
                {
                    Index idx = ((i1 + b * n) * k + i2) * m + i0;
                    dst_ref[idx] = src_val;
                }
            }
        }
    }
    // Launch tile operation
    scale_fiber<T>(alpha, src_tile, dst_tile, 0, batch);
    // Check results
    const Y eps = std::numeric_limits<Y>::epsilon();
    for(Index i = 0; i < dst_tile.nelems; ++i)
    {
        Y dst_val = Y{dst_tile[i]};
        Y dst_ref_val = dst_ref[i];
        REQUIRE(std::abs(dst_val - dst_ref_val) <= eps * std::abs(dst_ref_val));
    }
}

// Test cases
TEST_CASE("scale_fiber Tile", "[tile][scale_fiber]")
{
    // Test different sizes and alpha values
    test_tile<fp32_t>(1, 1, 1, 1, 1.0);
    test_tile<fp32_t>(2, 3, 4, 5, 2.0);
    test_tile<fp32_t>(3, 2, 1, 1, 0.5);
    test_tile<fp32_t>(1, 1, 1, 1, 0.0);
    test_tile<fp64_t>(1, 1, 1, 1, 1.0);
    test_tile<fp64_t>(2, 3, 4, 5, 2.0);
    test_tile<fp64_t>(3, 2, 1, 1, 0.5);
    test_tile<fp64_t>(1, 1, 1, 1, 0.0);
}