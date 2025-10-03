/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/relu_forward.cc
 * Placeholder for relu_forward kernel test
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/kernel/relu_forward.hh"

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
using namespace nntile::kernel::relu_forward;

// Type to acquire reference values
using ref_t = double;

// Struct to hold test data and reference results
template<typename T>
struct TestData
{
    using Y = typename T::repr_t;
    Index nelems; // Number of elements

    Y eps_check;

    std::vector<T> src;
    std::vector<T> dst_init;
    std::vector<T> dst_ref;
};

// Reference implementation of the forward ReLU operation
template<typename T>
void reference_relu_forward(TestData<T>& data)
{
    using Y = typename T::repr_t;
    if (data.nelems == 0)
    {
        return;
    }
    constexpr ref_t zero = 0.0;

    for(Index i = 0; i < data.nelems; ++i)
    {
        ref_t x = static_cast<Y>(data.src[i]);
        ref_t y = std::fmax(x, zero);
        data.dst_ref[i] = static_cast<T>(static_cast<Y>(y));
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

    switch(strategy)
    {
        // Non-random input generation
        case DataGen::PRESET:
            for(Index i = 0; i < data.nelems; ++i)
            {
                data.src[i] = Y(2 * i + 1 - data.nelems);
                data.dst_init[i] = Y(i - 1);
            }
            break;
        // Specific random initialization
        case DataGen::RANDOM:
            std::mt19937 gen(42);
            std::uniform_real_distribution<Y> dist(-2.0, 2.0);
            for(Index i = 0; i < data.nelems; ++i)
            {
                data.src[i] = dist(gen);
                data.dst_init[i] = dist(gen);
            }
    }
}

// Get test data and reference results
template<typename T>
TestData<T> get_test_data(
    Index nelems,
    DataGen strategy
)
{
    TestData<T> data;
    // Generate data by a provided strategy
    data.nelems = nelems;
    data.src.resize(nelems);
    data.dst_init.resize(nelems);
    data.dst_ref.resize(nelems);
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

    // Compute reference outputs
    reference_relu_forward(data);
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
    for(Index i = 0; i < data.nelems; ++i)
    {
        Y dst_ref = static_cast<Y>(data.dst_ref[i]);
        REQUIRE_THAT(
            static_cast<Y>(dst_out[i]),
            WithinRel(dst_ref, data.eps_check)
        );
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
            "[kernel][relu_forward][cpu][nelems=" +
            std::to_string(data.nelems) +
            "]"
        )
        {
            cpu<T>(
                data.nelems,
                &data.src[0],
                &dst_cpu[0]
            );
        };
    }
    else
    {
        cpu<T>(
            data.nelems,
            &data.src[0],
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
    CUDA_CHECK(cudaMalloc(&dev_src, sizeof(T) * data.nelems),
               "cudaMalloc dev_src");
    CUDA_CHECK(cudaMalloc(&dev_dst, sizeof(T) * data.nelems),
               "cudaMalloc dev_dst");

    std::vector<T> dst_cuda(data.dst_init);

    CUDA_CHECK(cudaMemcpy(dev_src, &data.src[0], sizeof(T) * data.nelems,
                          cudaMemcpyHostToDevice), "cudaMemcpy dev_src");
    CUDA_CHECK(cudaMemcpy(dev_dst, &dst_cuda[0], sizeof(T) * data.nelems,
                          cudaMemcpyHostToDevice), "cudaMemcpy dev_dst");

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream), "cudaStreamCreate");

    if constexpr (run_bench)
    {
        BENCHMARK(
            "[kernel][relu_forward][cuda][nelems=" +
            std::to_string(data.nelems) +
            "]"
        )
        {
            cuda<T>(
                stream,
                data.nelems,
                dev_src,
                dev_dst
            );
            cudaStreamSynchronize(stream);
        };
    }
    else
    {
        cuda<T>(
            stream,
            data.nelems,
            dev_src,
            dev_dst
        );
        CUDA_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

        CUDA_CHECK(cudaMemcpy(&dst_cuda[0], dev_dst, sizeof(T) * data.nelems,
                              cudaMemcpyDeviceToHost), "cudaMemcpy dst_cuda");

        verify_results(data, dst_cuda);
    }

    CUDA_CHECK(cudaFree(dev_src), "cudaFree dev_src");
    CUDA_CHECK(cudaFree(dev_dst), "cudaFree dev_dst");
    CUDA_CHECK(cudaStreamDestroy(stream), "cudaStreamDestroy");
}
#endif

// Catch2-based tests
TEMPLATE_TEST_CASE(
    "ReLU Forward Kernel Verification",
    "[relu_forward]",
    fp64_t,
    fp32_t,
    fp16_t,
    bf16_t
)
{
    using T = TestType;
    const Index nelems = GENERATE(5, 129);
    const DataGen strategy = GENERATE(DataGen::PRESET, DataGen::RANDOM);

    auto data = get_test_data<T>(
        nelems,
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
    "ReLU Forward Kernel Benchmark",
    "[relu_forward][!benchmark]",
    fp64_t,
    fp32_t,
    fp16_t,
    bf16_t
)
{
    using T = TestType;
    const Index nelems = GENERATE(512, 1024*1024);
    const DataGen strategy = GENERATE(DataGen::PRESET);

    auto data = get_test_data<T>(
        nelems,
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
