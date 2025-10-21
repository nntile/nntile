/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/scale_fiber.cc
 * Per-element scaling of a tensor with a broadcasted fiber
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/kernel/scale_fiber.hh"

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
using namespace nntile::kernel::scale_fiber;

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

// Test kernel on CPU
template<typename T>
void test_cpu(Index m, Index n, Index k, Index batch, Scalar alpha)
{
    // Prepare test data
    TestData<T> data(m, n, k, batch, alpha);
    // Launch CPU kernel
    cpu<T>(m, n, k, batch, alpha, data.src.data(), data.dst.data());
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

// Test kernel on CUDA
template<typename T>
void test_cuda(Index m, Index n, Index k, Index batch, Scalar alpha)
{
#ifdef NNTILE_USE_CUDA
    // Prepare test data
    TestData<T> data(m, n, k, batch, alpha);
    // Allocate CUDA memory
    T *src_gpu, *dst_gpu;
    CUDA_CHECK(cudaMalloc(&src_gpu, sizeof(T) * k * batch));
    CUDA_CHECK(cudaMalloc(&dst_gpu, sizeof(T) * m * k * n * batch));
    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(src_gpu, data.src.data(), sizeof(T) * k * batch,
            cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dst_gpu, data.dst.data(), sizeof(T) * m * k * n * batch,
            cudaMemcpyHostToDevice));
    // Get CUDA stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    // Launch CUDA kernel
    cuda<T>(stream, m, n, k, batch, alpha, src_gpu, dst_gpu);
    // Wait for completion
    CUDA_CHECK(cudaStreamSynchronize(stream));
    // Copy result back
    CUDA_CHECK(cudaMemcpy(data.dst.data(), dst_gpu, sizeof(T) * m * k * n * batch,
            cudaMemcpyDeviceToHost));
    // Check results
    using Y = typename T::repr_t;
    const Y eps = std::numeric_limits<Y>::epsilon();
    for(Index i = 0; i < m * k * n * batch; ++i)
    {
        Y dst_val = Y{data.dst[i]};
        Y dst_ref_val = data.dst_ref_ref[i];
        REQUIRE(std::abs(dst_val - dst_ref_val) <= eps * std::abs(dst_ref_val));
    }
    // Cleanup
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(src_gpu));
    CUDA_CHECK(cudaFree(dst_gpu));
#else
    // Skip CUDA test if not available
    REQUIRE(true);
#endif
}

// Test cases
TEST_CASE("scale_fiber CPU", "[kernel][scale_fiber]")
{
    // Test different sizes and alpha values
    test_cpu<fp32_t>(1, 1, 1, 1, 1.0);
    test_cpu<fp32_t>(2, 3, 4, 5, 2.0);
    test_cpu<fp32_t>(3, 2, 1, 1, 0.5);
    test_cpu<fp32_t>(1, 1, 1, 1, 0.0);
    test_cpu<fp64_t>(1, 1, 1, 1, 1.0);
    test_cpu<fp64_t>(2, 3, 4, 5, 2.0);
    test_cpu<fp64_t>(3, 2, 1, 1, 0.5);
    test_cpu<fp64_t>(1, 1, 1, 1, 0.0);
}

TEST_CASE("scale_fiber CUDA", "[kernel][scale_fiber]")
{
    // Test different sizes and alpha values
    test_cuda<fp32_t>(1, 1, 1, 1, 1.0);
    test_cuda<fp32_t>(2, 3, 4, 5, 2.0);
    test_cuda<fp32_t>(3, 2, 1, 1, 0.5);
    test_cuda<fp32_t>(1, 1, 1, 1, 0.0);
    test_cuda<fp64_t>(1, 1, 1, 1, 1.0);
    test_cuda<fp64_t>(2, 3, 4, 5, 2.0);
    test_cuda<fp64_t>(3, 2, 1, 1, 0.5);
    test_cuda<fp64_t>(1, 1, 1, 1, 0.0);
}