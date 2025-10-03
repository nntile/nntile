/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/norm_fiber.cc
 * Euclidean norms over slices into a fiber of a product of buffers
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/kernel/norm_fiber.hh"

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
using namespace nntile::kernel::norm_fiber;

// Type to acquire reference values
using ref_t = double;

// Struct to hold test data and reference results
template<typename T>
struct TestData
{
    using Y = typename T::repr_t;
    Index m, n, k, batch; // Dimensions
    Scalar alpha, beta;    // Scaling factors

    std::vector<T> src1;
    std::vector<T> src2;
    std::vector<T> dst_ref;
};

// Reference implementation of the norm_fiber operation
template<typename T>
void reference_norm_fiber(TestData<T>& data)
{
    using Y = typename T::repr_t;
    if (data.m == 0 || data.n == 0 || data.k == 0 || data.batch == 0)
    {
        return;
    }

    const ref_t alpha_r = data.alpha;
    const ref_t beta_r = data.beta;

    for(Index b = 0; b < data.batch; ++b)
    {
        for(Index i2 = 0; i2 < data.k; ++i2)
        {
            // Compute norm over m*n elements for this position
            ref_t norm_sq = 0.0;
            for(Index i1 = 0; i1 < data.n; ++i1)
            {
                for(Index i0 = 0; i0 < data.m; ++i0)
                {
                    Index src_idx = ((i1 + b * data.n) * data.k + i2) * data.m + i0;
                    ref_t val = static_cast<Y>(data.src1[src_idx]);
                    norm_sq += val * val;
                }
            }
            ref_t norm_val = std::sqrt(norm_sq) * alpha_r;
            ref_t src2_val = beta_r * static_cast<Y>(data.src2[i2 + b * data.k]);
            ref_t result = std::hypot(norm_val, src2_val);
            data.dst_ref[i2 + b * data.k] = static_cast<T>(static_cast<Y>(result));
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
    const Index total_elems = data.m * data.n * data.k * data.batch;

    data.src1.resize(total_elems);
    data.src2.resize(data.k * data.batch);
    data.dst_ref.resize(data.k * data.batch);

    switch(strategy)
    {
        // Non-random input generation
        case DataGen::PRESET:
            for(Index b = 0; b < data.batch; ++b)
            {
                for(Index i2 = 0; i2 < data.k; ++i2)
                {
                    data.src2[i2 + b * data.k] = Y{2.0 * i2 + 1.0};
                    for(Index i1 = 0; i1 < data.n; ++i1)
                    {
                        for(Index i0 = 0; i0 < data.m; ++i0)
                        {
                            Index src_idx = ((i1 + b * data.n) * data.k + i2) * data.m + i0;
                            data.src1[src_idx] = Y{1.0 + i0 + i1 * data.m + i2 * data.m * data.n};
                        }
                    }
                }
            }
            break;
        // Specific random initialization
        case DataGen::RANDOM:
            std::mt19937 gen(42);
            std::uniform_real_distribution<Y> dist(-2.0, 2.0);
            for(Index i = 0; i < total_elems; ++i)
            {
                data.src1[i] = dist(gen);
            }
            for(Index i = 0; i < data.k * data.batch; ++i)
            {
                data.src2[i] = dist(gen);
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
    Scalar beta,
    DataGen strategy
)
{
    TestData<T> data;
    data.m = m;
    data.n = n;
    data.k = k;
    data.batch = batch;
    data.alpha = alpha;
    data.beta = beta;

    // Generate data by a provided strategy
    generate_data(data, strategy);

    // Compute reference outputs
    reference_norm_fiber(data);
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
    // Set accuracy threshold for each precision
    Y eps_check;
    if (std::is_same_v<T, bf16_t>)
    {
        eps_check = 1e-1;
    }
    else if (std::is_same_v<T, fp16_t>)
    {
        eps_check = 1e-2;
    }
    else if (std::is_same_v<T, fp32_t>)
    {
        eps_check = 3.1e-3;
    }
    else if (std::is_same_v<T, fp64_t>)
    {
        eps_check = 1e-7;
    }
    else
    {
        throw std::runtime_error("Unsupported data type");
    }

    for(Index i = 0; i < data.dst_ref.size(); ++i)
    {
        Y ref = static_cast<Y>(data.dst_ref[i]);
        Y val = static_cast<Y>(dst_out[i]);
        REQUIRE_THAT(val, WithinRel(ref, eps_check) || WithinAbs(ref, eps_check));
    }
}

// Helper function to run CPU test and verify results
template<typename T, bool run_bench>
void run_cpu_test(TestData<T>& data)
{
    std::vector<T> dst_cpu(data.k * data.batch);

    if constexpr (run_bench)
    {
        BENCHMARK(
            "[kernel][norm_fiber][cpu][m=" +
            std::to_string(data.m) +
            "][n=" + std::to_string(data.n) +
            "][k=" + std::to_string(data.k) +
            "][batch=" + std::to_string(data.batch) +
            "]"
        )
        {
            cpu<T>(
                data.m,
                data.n,
                data.k,
                data.batch,
                data.alpha,
                &data.src1[0],
                data.beta,
                &data.src2[0],
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
            &data.src1[0],
            data.beta,
            &data.src2[0],
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
    T *dev_src1, *dev_src2, *dev_dst;
    CUDA_CHECK(cudaMalloc(&dev_src1, sizeof(T) * data.src1.size()),
               "cudaMalloc dev_src1");
    CUDA_CHECK(cudaMalloc(&dev_src2, sizeof(T) * data.src2.size()),
               "cudaMalloc dev_src2");
    CUDA_CHECK(cudaMalloc(&dev_dst, sizeof(T) * data.k * data.batch),
               "cudaMalloc dev_dst");

    std::vector<T> dst_cuda(data.k * data.batch);

    CUDA_CHECK(cudaMemcpy(dev_src1, &data.src1[0], sizeof(T) * data.src1.size(),
                          cudaMemcpyHostToDevice), "cudaMemcpy dev_src1");
    CUDA_CHECK(cudaMemcpy(dev_src2, &data.src2[0], sizeof(T) * data.src2.size(),
                          cudaMemcpyHostToDevice), "cudaMemcpy dev_src2");

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream), "cudaStreamCreate");

    if constexpr (run_bench)
    {
        BENCHMARK(
            "[kernel][norm_fiber][cuda][m=" +
            std::to_string(data.m) +
            "][n=" + std::to_string(data.n) +
            "][k=" + std::to_string(data.k) +
            "][batch=" + std::to_string(data.batch) +
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
                dev_src1,
                data.beta,
                dev_src2,
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
            dev_src1,
            data.beta,
            dev_src2,
            dev_dst
        );
        CUDA_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

        CUDA_CHECK(cudaMemcpy(&dst_cuda[0], dev_dst, sizeof(T) * data.k * data.batch,
                              cudaMemcpyDeviceToHost), "cudaMemcpy dst_cuda");

        verify_results(data, dst_cuda);
    }

    CUDA_CHECK(cudaFree(dev_src1), "cudaFree dev_src1");
    CUDA_CHECK(cudaFree(dev_src2), "cudaFree dev_src2");
    CUDA_CHECK(cudaFree(dev_dst), "cudaFree dev_dst");
    CUDA_CHECK(cudaStreamDestroy(stream), "cudaStreamDestroy");
}
#endif

// Catch2-based tests
TEMPLATE_TEST_CASE(
    "Norm Fiber Kernel Verification",
    "[norm_fiber]",
    fp64_t,
    fp32_t,
    fp16_t,
    bf16_t
)
{
    using T = TestType;
    const Index m = GENERATE(5, 32);
    const Index n = GENERATE(5, 32);
    const Index k = GENERATE(5, 16);
    const Index batch = GENERATE(1, 3);
    const Scalar alpha = GENERATE(0.5, 1.0, 2.0);
    const Scalar beta = GENERATE(0.0, 0.5, 1.0);
    const DataGen strategy = GENERATE(DataGen::PRESET, DataGen::RANDOM);

    auto data = get_test_data<T>(
        m,
        n,
        k,
        batch,
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
    "Norm Fiber Kernel Benchmark",
    "[norm_fiber][!benchmark]",
    fp64_t,
    fp32_t,
    fp16_t,
    bf16_t
)
{
    using T = TestType;
    const Index m = GENERATE(512, 1024);
    const Index n = GENERATE(512, 1024);
    const Index k = GENERATE(128, 256);
    const Index batch = GENERATE(1, 4);
    const Scalar alpha = GENERATE(1.0);
    const Scalar beta = GENERATE(1.0);
    const DataGen strategy = GENERATE(DataGen::PRESET);

    auto data = get_test_data<T>(
        m,
        n,
        k,
        batch,
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
