/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/norm_slice.cc
 * Euclidean norms of fibers into a slice of a buffer (out-of-place version)
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/kernel/norm_slice.hh"

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
using namespace nntile::kernel;
using namespace nntile::kernel::norm_slice;

// Type to acquire reference values
using ref_t = double;

// Import NNTile types
using nntile::fp32_t;
using nntile::fp64_t;

// Struct to hold test data and reference results
template<typename T>
struct TestData
{
    using Y = typename T::repr_t;
    // Sizes of source array
    Index m, n, k;
    // Scaling factors
    Scalar alpha, beta;
    // Input arrays
    std::vector<T> src1, src2;
    // Output array
    std::vector<T> dst;
    // Reference result
    std::vector<T> dst_ref;
};

// Function to generate random test data
template<typename T>
TestData<T> generate_test_data(Index m, Index n, Index k)
{
    // Initialize random generator
    std::mt19937 gen(42);
    using Y = typename T::repr_t;
    std::uniform_real_distribution<Y> dist(-1.0, 1.0);
    // Create test data
    TestData<T> data;
    data.m = m;
    data.n = n;
    data.k = k;
    data.alpha = static_cast<Scalar>(dist(gen));
    data.beta = static_cast<Scalar>(dist(gen));
    // Generate input arrays
    data.src1.resize(m * k * n);
    data.src2.resize(m * n);
    for(auto &x : data.src1)
    {
        x = T{dist(gen)};
    }
    for(auto &x : data.src2)
    {
        x = T{dist(gen)};
    }
    // Generate output array
    data.dst.resize(m * n);
    for(auto &x : data.dst)
    {
        x = T{dist(gen)};
    }
    // Compute reference result
    data.dst_ref.resize(m * n);
    for(Index i2 = 0; i2 < n; ++i2)
    {
        for(Index i1 = 0; i1 < m; ++i1)
        {
            // Compute norm of fiber src1[i1,:,i2]
            typename TestData<T>::Y norm_sq = typename TestData<T>::Y{0};
            for(Index i0 = 0; i0 < k; ++i0)
            {
                typename TestData<T>::Y val = typename TestData<T>::Y{data.src1[(i2 * k + i0) * m + i1]};
                norm_sq += val * val;
            }
            typename TestData<T>::Y fiber_norm = std::sqrt(norm_sq);
            // Combine with src2 and beta
            typename TestData<T>::Y src2_val = typename TestData<T>::Y{data.src2[i2 * m + i1]};
            typename TestData<T>::Y result_val = std::hypot(data.alpha * fiber_norm, data.beta * src2_val);
            data.dst_ref[i2 * m + i1] = T{result_val};
        }
    }
    return data;
}

// Test CPU version
TEMPLATE_TEST_CASE("kernel::norm_slice::cpu", "[kernel],[norm_slice],[cpu]",
        fp32_t, fp64_t)
{
    using T = TestType;
    // Test various sizes
    std::vector<std::tuple<Index, Index, Index>> sizes = {
        {1, 1, 1},
        {2, 3, 4},
        {5, 2, 3},
        {10, 10, 10}
    };
    for(auto [m, n, k] : sizes)
    {
        DYNAMIC_SECTION("m=" << m << ", n=" << n << ", k=" << k)
        {
            // Generate test data
            auto data = generate_test_data<T>(m, n, k);
            // Apply kernel
            cpu<T>(m, n, k, data.alpha, data.src1.data(), data.beta,
                    data.src2.data(), data.dst.data());
            // Check result
            for(Index i = 0; i < m * n; ++i)
            {
                REQUIRE_THAT(static_cast<ref_t>(data.dst[i].value),
                        WithinAbs(static_cast<ref_t>(data.dst_ref[i].value), 1e-5));
            }
        }
    }
}

// Test CUDA version (if available)
#ifdef NNTILE_USE_CUDA
TEMPLATE_TEST_CASE("kernel::norm_slice::cuda", "[kernel],[norm_slice],[cuda]",
        fp32_t, fp64_t)
{
    using T = TestType;
    // Test various sizes
    std::vector<std::tuple<Index, Index, Index>> sizes = {
        {1, 1, 1},
        {2, 3, 4},
        {5, 2, 3},
        {10, 10, 10}
    };
    for(auto [m, n, k] : sizes)
    {
        DYNAMIC_SECTION("m=" << m << ", n=" << n << ", k=" << k)
        {
            // Generate test data
            auto data = generate_test_data<T>(m, n, k);
            // Copy data to GPU
            cudaStream_t stream = cudaStreamDefault;
            T *src1_gpu, *src2_gpu, *dst_gpu;
            cudaMalloc(&src1_gpu, data.src1.size() * sizeof(T));
            cudaMalloc(&src2_gpu, data.src2.size() * sizeof(T));
            cudaMalloc(&dst_gpu, data.dst.size() * sizeof(T));
            cudaMemcpy(src1_gpu, data.src1.data(), data.src1.size() * sizeof(T),
                    cudaMemcpyHostToDevice);
            cudaMemcpy(src2_gpu, data.src2.data(), data.src2.size() * sizeof(T),
                    cudaMemcpyHostToDevice);
            cudaMemcpy(dst_gpu, data.dst.data(), data.dst.size() * sizeof(T),
                    cudaMemcpyHostToDevice);
            // Apply kernel
            cuda<T>(stream, m, n, k, data.alpha, src1_gpu, data.beta,
                    src2_gpu, dst_gpu);
            cudaStreamSynchronize(stream);
            // Copy result back
            cudaMemcpy(data.dst.data(), dst_gpu, data.dst.size() * sizeof(T),
                    cudaMemcpyDeviceToHost);
            // Check result
            for(Index i = 0; i < m * n; ++i)
            {
                REQUIRE_THAT(static_cast<ref_t>(data.dst[i].value),
                        WithinAbs(static_cast<ref_t>(data.dst_ref[i].value), 1e-4));
            }
            // Free GPU memory
            cudaFree(src1_gpu);
            cudaFree(src2_gpu);
            cudaFree(dst_gpu);
        }
    }
}
#endif // NNTILE_USE_CUDA
