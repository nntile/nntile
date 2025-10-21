/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/scale_fiber.cc
 * Per-element scaling of a tensor with a broadcasted fiber
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
// CUDA_CHECK definition
#include <nntile/kernel/cuda.hh>

// Use namespaces for shorter code
using namespace Catch;
using namespace Catch::Matchers;

// Use tested NNTile namespaces
using namespace nntile;
using namespace nntile::tile;

// Type to acquire reference values
using ref_t = double;

// Struct to hold test data and reference results
template<typename T>
struct TestData
{
    using Y = typename T::repr_t;
    Index m, n, k, batch; // Tensor dimensions
    Scalar alpha;         // Scalar factor

    Y eps_check;

    std::vector<T> src_init;
    std::vector<T> dst_init;

    std::vector<T> dst_ref;
};

// Reference implementation of the scale fiber operation
template<typename T>
void reference_scale_fiber(TestData<T>& data)
{
    using Y = typename T::repr_t;

    data.dst_ref = data.dst_init; // Copy initial destination

    for(Index b = 0; b < data.batch; ++b)
    {
        for(Index i2 = 0; i2 < data.k; ++i2)
        {
            Index src_idx = i2 + b * data.k;
            ref_t src_val = static_cast<Y>(data.src_init[src_idx]);
            src_val *= data.alpha;

            for(Index i1 = 0; i1 < data.n; ++i1)
            {
                Index dst_idx_base = (i1 + b * data.n) * data.k + i2;
                for(Index i0 = 0; i0 < data.m; ++i0)
                {
                    Index dst_idx = dst_idx_base * data.m + i0;
                    data.dst_ref[dst_idx] = static_cast<Y>(src_val);
                }
            }
        }
    }
}

// Enum for data generation strategies
enum class DataGen
{
    PRESET,
    RANDOM
};

// Generates data with preset, deterministic values
template<typename T>
void generate_data(TestData<T>& data, DataGen strategy)
{
    using Y = typename T::repr_t;

    Index num_elems = data.m * data.n * data.k * data.batch;

    data.src_init.resize(data.k * data.batch);
    data.dst_init.resize(num_elems);
    data.dst_ref.resize(num_elems);

    switch(strategy)
    {
        // Non-random input generation
        case DataGen::PRESET:
            for(Index i = 0; i < data.k * data.batch; ++i)
            {
                Index tmp_i = 2 * i + 1 - data.k * data.batch;
                data.src_init[i] = static_cast<Y>(tmp_i);
            }
            for(Index i = 0; i < num_elems; ++i)
            {
                Index tmp_i = 5 * num_elems - 2 * i;
                data.dst_init[i] = static_cast<Y>(tmp_i);
            }
            break;
        // Specific random initialization
        case DataGen::RANDOM:
            std::mt19937 gen(42);
            std::uniform_real_distribution<Y> dist(-2.0, 2.0);
            for(Index i = 0; i < data.k * data.batch; ++i)
            {
                data.src_init[i] = dist(gen);
            }
            for(Index i = 0; i < num_elems; ++i)
            {
                data.dst_init[i] = dist(gen);
            }
            break;
    }
}

// Get test data and reference results
template<typename T>
TestData<T> get_test_data(
    Index m,
    Index n,
    Index k,
    Index batch,
    Scalar alpha,
    DataGen strategy
)
{
    using Y = typename T::repr_t;
    TestData<T> data;
    data.m = m;
    data.n = n;
    data.k = k;
    data.batch = batch;
    data.alpha = alpha;

    // Generate data by a provided strategy
    generate_data(data, strategy);

    // Set accuracy threshold for each precision
    if (std::is_same_v<T, bf16_t>)
    {
        data.eps_check = Y{1e-1};
    }
    else if (std::is_same_v<T, fp16_t>)
    {
        data.eps_check = Y{1e-2};
    }
    else if (std::is_same_v<T, fp32_t>)
    {
        data.eps_check = Y{1e-6};
    }
    else if (std::is_same_v<T, fp64_t>)
    {
        data.eps_check = Y{1e-12};
    }
    else
    {
        throw std::runtime_error("Unsupported data type");
    }

    // Compute reference outputs
    reference_scale_fiber(data);
    return data;
}

// Helper function to verify results
template<typename T>
void verify_results(
    const TestData<T>& data,
    const std::vector<T>& src,
    const std::vector<T>& dst)
{
    using Y = typename T::repr_t;

    // Check that source data was not modified
    for(Index i = 0; i < data.k * data.batch; ++i)
    {
        REQUIRE(static_cast<Y>(src[i]) == static_cast<Y>(data.src_init[i]));
    }

    // Check output
    for(Index i = 0; i < data.m * data.n * data.k * data.batch; ++i)
    {
        REQUIRE_THAT(
            static_cast<Y>(dst[i]),
            WithinRel(static_cast<Y>(data.dst_ref[i]), data.eps_check)
        );
    }
}

// Helper function to run Tile test and verify results
template<typename T, bool run_bench>
void run_tile_test(TestData<T>& data)
{
    std::vector<T> dst_tile(data.dst_init);
    std::vector<T> src_tile(data.src_init);

    if constexpr (run_bench)
    {
        BENCHMARK(
            "[tile][scale_fiber][m=" +
            std::to_string(data.m) +
            "][n=" + std::to_string(data.n) +
            "][k=" + std::to_string(data.k) +
            "][batch=" + std::to_string(data.batch) +
            "][alpha=" + std::to_string(data.alpha) + "]"
        )
        {
            // Placeholder for tile test
            scale_fiber_async<T>(data.alpha, src_tile, dst_tile, 0, data.batch);
        };
    }
    else
    {
        // Placeholder for tile test
        scale_fiber_async<T>(data.alpha, src_tile, dst_tile, 0, data.batch);
        verify_results(data, src_tile, dst_tile);
    }
}

// Catch2-based tests
TEMPLATE_TEST_CASE(
    "Scale Fiber Tile Verification",
    "[scale_fiber]",
    fp64_t,
    fp32_t,
    fp16_t,
    bf16_t
)
{
    using T = TestType;
    const Index m = GENERATE(1, 5);
    const Index n = GENERATE(1, 3);
    const Index k = GENERATE(1, 10);
    const Index batch = GENERATE(1, 4);
    const Scalar alpha = GENERATE(0.5, 1.0, 2.0);
    const DataGen strategy = GENERATE(DataGen::PRESET, DataGen::RANDOM);

    auto data = get_test_data<T>(m, n, k, batch, alpha, strategy);

    SECTION("tile")
    {
        run_tile_test<T, false>(data);
    }
}

// Catch2-based benchmarks
TEMPLATE_TEST_CASE(
    "Scale Fiber Tile Benchmark",
    "[scale_fiber][!benchmark]",
    fp64_t,
    fp32_t,
    fp16_t,
    bf16_t
)
{
    using T = TestType;
    const Index m = GENERATE(64, 256);
    const Index n = GENERATE(64, 256);
    const Index k = GENERATE(32, 128);
    const Index batch = GENERATE(4, 16);
    const Scalar alpha = GENERATE(1.0);
    const DataGen strategy = GENERATE(DataGen::PRESET);

    auto data = get_test_data<T>(m, n, k, batch, alpha, strategy);

    SECTION("tile")
    {
        run_tile_test<T, true>(data);
    }
}
