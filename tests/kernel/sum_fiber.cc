/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/sum_fiber.cc
 * Sums over slices into a fiber of a buffer
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/kernel/sum_fiber.hh"

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

// Use namespaces for shorter code
using namespace Catch;

// Use tested NNTile namespaces
using namespace nntile;
using namespace nntile::kernel;
using namespace nntile::kernel::sum_fiber;

// Type to acquire reference values
using ref_t = double;

// Struct to hold test data and reference results
template<typename T>
struct TestData
{
    using Y = typename T::repr_t;
    Index m;     // First mode size
    Index n;     // Last mode size
    Index k;     // Middle mode size
    Index batch; // Batch dimension
    Scalar alpha;
    Scalar beta;
    Scalar eps_check;

    std::vector<T> src;      // Size: m*k*n*batch
    std::vector<T> dst_init; // Size: k*batch
    std::vector<T> dst_ref;  // Size: k*batch
};

// Reference implementation of the sum_fiber operation
template<typename T>
void reference_sum_fiber(TestData<T>& data)
{
    using Y = typename T::repr_t;
    const ref_t alpha = data.alpha;
    const ref_t beta = data.beta;
    const ref_t zero = 0.0;

    // Cycle over batch
    for(Index b = 0; b < data.batch; ++b)
    {
        // Cycle over the only axis of output buffer
        for(Index i2 = 0; i2 < data.k; ++i2)
        {
            // Init sum
            ref_t sum = zero;
            // Cycle over the third axis of input buffer
            for(Index i1 = 0; i1 < data.n; ++i1)
            {
                // Cycle over the first axis of input buffer
                for(Index i0 = 0; i0 < data.m; ++i0)
                {
                    // Read value from source
                    Index src_idx = ((i1 + b * data.n) * data.k + i2) * data.m + i0;
                    ref_t val = static_cast<Y>(data.src[src_idx]);
                    // Update sum
                    sum += val;
                }
            }
            // Save result
            ref_t dst_val;
            if(beta == zero)
            {
                dst_val = alpha * sum;
            }
            else
            {
                dst_val = beta * static_cast<Y>(data.dst_init[i2 + b * data.k]) + alpha * sum;
            }
            data.dst_ref[i2 + b * data.k] = static_cast<T>(static_cast<Y>(dst_val));
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
void generate_data(TestData<T>& data, Index m, Index n, Index k, Index batch, DataGen strategy)
{
    using Y = typename T::repr_t;
    data.m = m;
    data.n = n;
    data.k = k;
    data.batch = batch;

    data.src.resize(m * k * n * batch);
    data.dst_init.resize(k * batch);
    data.dst_ref.resize(k * batch);

    switch(strategy)
    {
        // Non-random input generation
        case DataGen::PRESET:
            for(Index i = 0; i < m * k * n * batch; ++i)
            {
                data.src[i] = Y(i % 10 - 5);
            }
            for(Index i = 0; i < k * batch; ++i)
            {
                data.dst_init[i] = Y(i + 1);
            }
            break;
        // Specific random initialization
        case DataGen::RANDOM:
            std::mt19937 gen(42);
            std::uniform_real_distribution<Y> dist_src(-2.0, 2.0);
            std::uniform_real_distribution<Y> dist_dst(0.0, 5.0);
            for(Index i = 0; i < m * k * n * batch; ++i)
            {
                data.src[i] = dist_src(gen);
            }
            for(Index i = 0; i < k * batch; ++i)
            {
                data.dst_init[i] = dist_dst(gen);
            }
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
    Scalar beta,
    DataGen strategy
)
{
    TestData<T> data;
    // Generate data by a provided strategy
    generate_data(data, m, n, k, batch, strategy);
    // Fill in remaining fields of TestData
    data.alpha = alpha;
    data.beta = beta;
    // Set accuracy threshold for each precision
    if (std::is_same_v<T, bf16_t>)
    {
        data.eps_check = 1e-1;
    }
    else if (std::is_same_v<T, fp16_t>)
    {
        data.eps_check = 1e-2;
    }
    else if (std::is_same_v<T, fp32_t>)
    {
        data.eps_check = 1e-4;
    }
    else if (std::is_same_v<T, fp64_t>)
    {
        data.eps_check = 1e-10;
    }
    else
    {
        throw std::runtime_error("Unsupported data type");
    }
    // Compute reference outputs
    data.dst_ref = data.dst_init;
    reference_sum_fiber(data);
    return data;
}

// Helper function to verify results
template<typename T>
void verify_results(
    const TestData<T>& data,
    const std::vector<T>& dst_out
)
{
    using Y = typename T::repr_t;
    for(Index i = 0; i < data.k * data.batch; ++i)
    {
        Y dst_ref = static_cast<Y>(data.dst_ref[i]);
        auto dst_approx = Approx(dst_ref).epsilon(data.eps_check);
        REQUIRE(static_cast<Y>(dst_out[i]) == dst_approx);
    }
}

// Helper function to run CPU test and verify results
template<typename T, bool run_bench>
void run_cpu_test(TestData<T>& data)
{
    std::vector<T> dst_cpu(data.dst_init);

    if constexpr (run_bench)
    {
        BENCHMARK(
            "[kernel][sum_fiber][cpu][m=" +
            std::to_string(data.m) +
            "][n=" +
            std::to_string(data.n) +
            "][k=" +
            std::to_string(data.k) +
            "][batch=" +
            std::to_string(data.batch) +
            "]"
        )
        {
            cpu<T>(
                data.m,
                data.n,
                data.k,
                data.batch,
                data.alpha,
                &data.src[0],
                data.beta,
                &dst_cpu[0]
            );
        };
    }
    else
    {
        cpu<T>(
            data.m,
            data.n,
            data.k,
            data.batch,
            data.alpha,
            &data.src[0],
            data.beta,
            &dst_cpu[0]
        );
        verify_results(data, dst_cpu);
    }
}

#ifdef NNTILE_USE_CUDA

// Helper function to run CUDA test and verify results
template<typename T, bool run_bench>
void run_cuda_test(TestData<T>& data)
{
    T *dev_src, *dev_dst;
    CUDA_CHECK(cudaMalloc(&dev_src, sizeof(T) * data.m * data.k * data.n * data.batch),
               "cudaMalloc dev_src");
    CUDA_CHECK(cudaMalloc(&dev_dst, sizeof(T) * data.k * data.batch),
               "cudaMalloc dev_dst");

    std::vector<T> dst_cuda(data.dst_init);

    CUDA_CHECK(cudaMemcpy(dev_src, &data.src[0],
                          sizeof(T) * data.m * data.k * data.n * data.batch,
                          cudaMemcpyHostToDevice), "cudaMemcpy dev_src");
    CUDA_CHECK(cudaMemcpy(dev_dst, &dst_cuda[0],
                          sizeof(T) * data.k * data.batch,
                          cudaMemcpyHostToDevice), "cudaMemcpy dev_dst");

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream), "cudaStreamCreate");

    if constexpr (run_bench)
    {
        BENCHMARK(
            "[kernel][sum_fiber][cuda][m=" +
            std::to_string(data.m) +
            "][n=" +
            std::to_string(data.n) +
            "][k=" +
            std::to_string(data.k) +
            "][batch=" +
            std::to_string(data.batch) +
            "]"
        )
        {
            cuda<T>(
                stream,
                data.m,
                data.n,
                data.k,
                data.batch,
                data.alpha,
                dev_src,
                data.beta,
                dev_dst
            );
            cudaStreamSynchronize(stream);
        };
    }
    else
    {
        cuda<T>(
            stream,
            data.m,
            data.n,
            data.k,
            data.batch,
            data.alpha,
            dev_src,
            data.beta,
            dev_dst
        );
        CUDA_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

        CUDA_CHECK(cudaMemcpy(&dst_cuda[0], dev_dst,
                              sizeof(T) * data.k * data.batch,
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy dst_cuda");

        verify_results(data, dst_cuda);
    }

    CUDA_CHECK(cudaFree(dev_src), "cudaFree dev_src");
    CUDA_CHECK(cudaFree(dev_dst), "cudaFree dev_dst");
    CUDA_CHECK(cudaStreamDestroy(stream), "cudaStreamDestroy");
}
#endif

// Catch2-based tests
TEMPLATE_TEST_CASE(
    "Sum Fiber Kernel Verification",
    "[sum_fiber]",
    fp64_t,
    fp32_t,
    fp16_t,
    bf16_t
)
{
    using T = TestType;
    const Index m = GENERATE(3, 8);
    const Index n = GENERATE(2, 5);
    const Index k = GENERATE(4, 10);
    const Index batch = GENERATE(1, 2);
    const Scalar alpha = GENERATE(1.0, 2.0);
    const Scalar beta = GENERATE(0.0, 1.0);
    const DataGen strategy = GENERATE(DataGen::PRESET, DataGen::RANDOM);

    auto data = get_test_data<T>(
        m, n, k, batch,
        alpha,
        beta,
        strategy
    );

    SECTION("cpu")
    {
        run_cpu_test<T, false>(data);
    }

#ifdef NNTILE_USE_CUDA
    SECTION("cuda")
    {
        run_cuda_test<T, false>(data);
    }
#endif
}

// Catch2-based benchmarks
TEMPLATE_TEST_CASE(
    "Sum Fiber Kernel Benchmark",
    "[sum_fiber][!benchmark]",
    fp64_t,
    fp32_t,
    fp16_t,
    bf16_t
)
{
    using T = TestType;
    const Index m = GENERATE(64);
    const Index n = GENERATE(64);
    const Index k = GENERATE(64);
    const Index batch = GENERATE(4);
    const Scalar alpha = GENERATE(1.0);
    const Scalar beta = GENERATE(0.0);
    const DataGen strategy = GENERATE(DataGen::PRESET);

    auto data = get_test_data<T>(
        m, n, k, batch,
        alpha,
        beta,
        strategy
    );

    SECTION("cpu")
    {
        run_cpu_test<T, true>(data);
    }

#ifdef NNTILE_USE_CUDA
    SECTION("cuda")
    {
        run_cuda_test<T, true>(data);
    }
#endif
}
