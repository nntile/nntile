/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/multiply_fiber.cc
 * Per-element multiplication of a tensor by a broadcasted fiber
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/kernel/multiply_fiber.hh"

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
using namespace nntile::kernel::multiply_fiber;

// Type to acquire reference values
using ref_t = double;

// Struct to hold test data and reference results
template<typename T>
struct TestData
{
    using Y = typename T::repr_t;
    Index m, n, k; // Dimensions
    Scalar alpha;  // Scaling factor

    Y eps_check;

    std::vector<T> src1_init;
    std::vector<T> src2_init;
    std::vector<T> dst_init;

    std::vector<T> dst_ref;
};

// Reference implementation of the multiply_fiber operation
template<typename T>
void reference_multiply_fiber(TestData<T>& data)
{
    using Y = typename T::repr_t;
    if (data.m == 0 || data.n == 0 || data.k == 0)
    {
        return;
    }

    const ref_t alpha_r = data.alpha;

    for(Index i2 = 0; i2 < data.k; ++i2)
    {
        ref_t src1_val = static_cast<Y>(data.src1_init[i2]);
        src1_val *= alpha_r;
        for(Index i1 = 0; i1 < data.n; ++i1)
        {
            for(Index i0 = 0; i0 < data.m; ++i0)
            {
                Index src2_idx = (i1 * data.k + i2) * data.m + i0;
                ref_t src2_val = static_cast<Y>(data.src2_init[src2_idx]);
                data.dst_ref[src2_idx] = static_cast<Y>(src1_val * src2_val);
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

    data.src1_init.resize(data.k);
    data.src2_init.resize(data.m * data.n * data.k);
    data.dst_init.resize(data.m * data.n * data.k);
    data.dst_ref.resize(data.m * data.n * data.k);

    switch(strategy)
    {
        // Non-random input generation
        case DataGen::PRESET:
            for(Index i2 = 0; i2 < data.k; ++i2)
            {
                Y src1_val = 2.0 * i2 + 1.0;
                data.src1_init[i2] = src1_val;
            }
            for(Index i2 = 0; i2 < data.k; ++i2)
            {
                for(Index i1 = 0; i1 < data.n; ++i1)
                {
                    for(Index i0 = 0; i0 < data.m; ++i0)
                    {
                        Index src2_idx = (i1 * data.k + i2) * data.m + i0;
                        Y src2_val = 1.0 + i0 + i1 * data.m +
                            i2 * data.m * data.n;
                        data.src2_init[src2_idx] = src2_val;
                    }
                }
            }
            // Initialize dst_init with some values
            for(Index i = 0; i < data.m * data.n * data.k; ++i)
            {
                data.dst_init[i] = static_cast<Y>(-1.0);
            }
            break;
        // Specific random initialization
        case DataGen::RANDOM:
            std::mt19937 gen(42);
            std::uniform_real_distribution<Y> dist(-2.0, 2.0);
            for(Index i = 0; i < data.src1_init.size(); ++i)
            {
                data.src1_init[i] = dist(gen);
            }
            for(Index i = 0; i < data.src2_init.size(); ++i)
            {
                data.src2_init[i] = dist(gen);
                data.dst_init[i] = dist(gen);
            }
            break;
    }
}

// Get test input data (reference computation is done separately)
template<typename T>
TestData<T> get_test_input_data(
    Index m,
    Index n,
    Index k,
    Scalar alpha,
    DataGen strategy
)
{
    TestData<T> data;
    data.m = m;
    data.n = n;
    data.k = k;
    data.alpha = alpha;

    // Generate data by a provided strategy
    generate_data(data, strategy);

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
        data.eps_check = 3.1e-3;
    }
    else if (std::is_same_v<T, fp64_t>)
    {
        data.eps_check = 1e-7;
    }
    else
    {
        throw std::runtime_error("Unsupported data type");
    }

    return data;
}

// Helper function to verify results
template<typename T>
void verify_results(
    const TestData<T>& data,
    const std::vector<T>& src1,
    const std::vector<T>& src2,
    const std::vector<T>& dst
)
{
    using Y = typename T::repr_t;

    // Check that src1 and src2 were not changed during kernel execution
    for(Index i = 0; i < data.src1_init.size(); ++i)
    {
        REQUIRE(static_cast<Y>(src1[i]) == static_cast<Y>(data.src1_init[i]));
    }
    for(Index i = 0; i < data.src2_init.size(); ++i)
    {
        REQUIRE(static_cast<Y>(src2[i]) == static_cast<Y>(data.src2_init[i]));
    }

    // Check that dst (output) matches reference
    for(Index i = 0; i < data.dst_ref.size(); ++i)
    {
        Y ref = static_cast<Y>(data.dst_ref[i]);
        REQUIRE_THAT(
            static_cast<Y>(dst[i]),
            WithinRel(ref, data.eps_check)
        );
    }
}

// Helper function to run CPU test and verify results
template<typename T, bool run_bench>
void run_cpu_test(TestData<T>& data)
{
    std::vector<T> dst_cpu(data.dst_init);
    std::vector<T> src1_cpu(data.src1_init);
    std::vector<T> src2_cpu(data.src2_init);

    if constexpr (run_bench)
    {
        BENCHMARK(
            "[kernel][multiply_fiber][cpu][m=" +
            std::to_string(data.m) +
            "][n=" + std::to_string(data.n) +
            "][k=" + std::to_string(data.k) +
            "]"
        )
        {
            cpu<T>(
                data.m,
                data.n,
                data.k,
                data.alpha,
                &src1_cpu[0],
                &src2_cpu[0],
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
            data.alpha,
            &src1_cpu[0],
            &src2_cpu[0],
            &dst_cpu[0]
        );
        verify_results(data, src1_cpu, src2_cpu, dst_cpu);
    }
}

#ifdef NNTILE_USE_CUDA

// Helper function to run CUDA test and verify results
template<typename T, bool run_bench>
void run_cuda_test(TestData<T>& data)
{
    T *dev_src1, *dev_src2, *dev_dst;
    CUDA_CHECK(
        cudaMalloc(&dev_src1, sizeof(T) * data.src1_init.size()),
        "cudaMalloc dev_src1"
    );
    CUDA_CHECK(
        cudaMalloc(&dev_src2, sizeof(T) * data.src2_init.size()),
        "cudaMalloc dev_src2"
    );
    CUDA_CHECK(
        cudaMalloc(&dev_dst, sizeof(T) * data.dst_ref.size()),
        "cudaMalloc dev_dst"
    );

    std::vector<T> dst_cuda(data.dst_init);
    std::vector<T> src1_cuda(data.src1_init);
    std::vector<T> src2_cuda(data.src2_init);

    CUDA_CHECK(
        cudaMemcpy(
            dev_src1,
            &src1_cuda[0],
            sizeof(T) * data.src1_init.size(),
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy dev_src1"
    );
    CUDA_CHECK(
        cudaMemcpy(
            dev_src2,
            &src2_cuda[0],
            sizeof(T) * data.src2_init.size(),
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy dev_src2"
    );
    CUDA_CHECK(
        cudaMemcpy(
            dev_dst,
            &dst_cuda[0],
            sizeof(T) * data.dst_init.size(),
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy dev_dst"
    );

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream), "cudaStreamCreate");

    if constexpr (run_bench)
    {
        BENCHMARK(
            "[kernel][multiply_fiber][cuda][m=" +
            std::to_string(data.m) +
            "][n=" + std::to_string(data.n) +
            "][k=" + std::to_string(data.k) +
            "]"
        )
        {
            cuda<T>(
                stream,
                data.m,
                data.n,
                data.k,
                data.alpha,
                dev_src1,
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
            data.alpha,
            dev_src1,
            dev_src2,
            dev_dst
        );
        CUDA_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

        CUDA_CHECK(
            cudaMemcpy(
                &dst_cuda[0],
                dev_dst,
                sizeof(T) * data.dst_init.size(),
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy dst_cuda"
        );
        CUDA_CHECK(
            cudaMemcpy(
                &src1_cuda[0],
                dev_src1,
                sizeof(T) * data.src1_init.size(),
                cudaMemcpyDeviceToHost
            ), "cudaMemcpy src1_cuda"
        );
        CUDA_CHECK(
            cudaMemcpy(
                &src2_cuda[0],
                dev_src2,
                sizeof(T) * data.src2_init.size(),
                cudaMemcpyDeviceToHost
            ), "cudaMemcpy src2_cuda"
        );

        verify_results(data, src1_cuda, src2_cuda, dst_cuda);
    }

    CUDA_CHECK(cudaFree(dev_src1), "cudaFree dev_src1");
    CUDA_CHECK(cudaFree(dev_src2), "cudaFree dev_src2");
    CUDA_CHECK(cudaFree(dev_dst), "cudaFree dev_dst");
    CUDA_CHECK(cudaStreamDestroy(stream), "cudaStreamDestroy");
}
#endif

// Catch2-based tests
TEMPLATE_TEST_CASE(
    "Prod Fiber3 Kernel Verification",
    "[multiply_fiber]",
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
    const Scalar alpha = GENERATE(0.5, 1.0, 2.0);
    const DataGen strategy = GENERATE(DataGen::PRESET, DataGen::RANDOM);

    auto data = get_test_input_data<T>(
        m,
        n,
        k,
        alpha,
        strategy
    );

    // Compute reference outputs for verification
    reference_multiply_fiber(data);

    SECTION(("cpu")
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
    "Prod Fiber3 Kernel Benchmark",
    "[multiply_fiber][!benchmark]",
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
    const Scalar alpha = GENERATE(1.0);
    const DataGen strategy = GENERATE(DataGen::PRESET);

    auto data = get_test_input_data<T>(
        m,
        n,
        k,
        alpha,
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
