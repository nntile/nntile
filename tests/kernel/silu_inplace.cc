/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/silu_inplace.cc
 * Inplace SiLU kernel test
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/kernel/silu_inplace.hh"

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
using namespace nntile::kernel::silu_inplace;

// Type to acquire reference values
using ref_t = double;

// Struct to hold test data and reference results
template<typename T>
struct TestData
{
    using Y = typename T::repr_t;
    Index nelems; // Number of elements

    Y eps_check;

    std::vector<T> data_init;
    std::vector<T> data_ref;
};

// Reference implementation of the inplace SiLU operation
template<typename T>
void reference_silu_inplace(TestData<T>& data)
{
    using Y = typename T::repr_t;
    if (data.nelems == 0)
    {
        return;
    }

    constexpr ref_t one = 1.0;

    for(Index i = 0; i < data.nelems; ++i)
    {
        ref_t x = static_cast<Y>(data.data_init[i]);
        ref_t y = x / (one + std::exp(-x));
        data.data_ref[i] = static_cast<T>(static_cast<Y>(y));
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
                data.data_init[i] = Y(2 * i + 1 - data.nelems);
            }
            break;
        // Specific random initialization
        case DataGen::RANDOM:
            std::mt19937 gen(42);
            std::uniform_real_distribution<Y> dist(-3.0, 3.0);
            for(Index i = 0; i < data.nelems; ++i)
            {
                data.data_init[i] = dist(gen);
            }
    }
}

// Get test input data (reference computation is done separately)
template<typename T>
TestData<T> get_test_input_data(
    Index nelems,
    DataGen strategy
)
{
    TestData<T> data;
    // Generate data by a provided strategy
    data.nelems = nelems;
    data.data_init.resize(nelems);
    data.data_ref.resize(nelems);
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
    const std::vector<T>& data_result
)
{
    using Y = typename T::repr_t;

    for(Index i = 0; i < data.nelems; ++i)
    {
        Y data_ref = static_cast<Y>(data.data_ref[i]);
        REQUIRE_THAT(
            static_cast<Y>(data_result[i]),
            WithinRel(data_ref, data.eps_check) ||
            WithinAbs(data_ref, data.eps_check)
        );
    }
}

// Helper function to run CPU test and verify results
template<typename T, bool run_bench>
void run_cpu_test(TestData<T>& data)
{
    std::vector<T> data_cpu(data.data_init);

    if constexpr (run_bench)
    {
        BENCHMARK(
            "[kernel][silu_inplace][cpu][nelems=" +
            std::to_string(data.nelems) +
            "]"
        )
        {
            cpu<T>(
                data.nelems,
                &data_cpu[0]
            );
        };
    }
    else
    {
        cpu<T>(
            data.nelems,
            &data_cpu[0]
        );
        verify_results(data, data_cpu);
    }
}

#ifdef NNTILE_USE_CUDA

// Helper function to run CUDA test and verify results
template<typename T, bool run_bench>
void run_cuda_test(TestData<T>& data)
{
    T *dev_data;
    CUDA_CHECK(cudaMalloc(&dev_data, sizeof(T) * data.nelems),
               "cudaMalloc dev_data");

    std::vector<T> data_cuda(data.data_init);

    CUDA_CHECK(cudaMemcpy(dev_data, &data_cuda[0], sizeof(T) * data.nelems,
                          cudaMemcpyHostToDevice), "cudaMemcpy dev_data");

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream), "cudaStreamCreate");

    if constexpr (run_bench)
    {
        BENCHMARK(
            "[kernel][silu_inplace][cuda][nelems=" +
            std::to_string(data.nelems) +
            "]"
        )
        {
            cuda<T>(
                stream,
                data.nelems,
                dev_data
            );
            cudaStreamSynchronize(stream);
        };
    }
    else
    {
        cuda<T>(
            stream,
            data.nelems,
            dev_data
        );
        CUDA_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

        CUDA_CHECK(cudaMemcpy(&data_cuda[0], dev_data, sizeof(T) * data.nelems,
                              cudaMemcpyDeviceToHost), "cudaMemcpy data_cuda");

        verify_results(data, data_cuda);
    }

    CUDA_CHECK(cudaFree(dev_data), "cudaFree dev_data");
    CUDA_CHECK(cudaStreamDestroy(stream), "cudaStreamDestroy");
}
#endif

// Catch2-based tests
TEMPLATE_TEST_CASE(
    "Inplace SiLU Kernel Verification",
    "[silu_inplace]",
    fp64_t,
    fp32_t,
    fp16_t,
    bf16_t
)
{
    using T = TestType;
    const Index nelems = GENERATE(5, 129);
    const DataGen strategy = GENERATE(DataGen::PRESET, DataGen::RANDOM);

    auto data = get_test_input_data<T>(
        nelems,
        strategy
    );

    // Compute reference outputs for verification
    reference_silu_inplace(data);

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
    "Inplace SiLU Kernel Benchmark",
    "[silu_inplace][!benchmark]",
    fp64_t,
    fp32_t,
    fp16_t,
    bf16_t
)
{
    using T = TestType;
    const Index nelems = GENERATE(512, 1024*1024);
    const DataGen strategy = GENERATE(DataGen::PRESET);

    auto data = get_test_input_data<T>(
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
