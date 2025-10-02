/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/add_scalar.cc
 * Add scalar operation on buffer
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/kernel/add_scalar.hh"

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
using namespace nntile::kernel::add_scalar;

// Type to acquire reference values
using ref_t = double;

// Struct to hold test data and reference results
template<typename T>
struct TestData
{
    using Y = typename T::repr_t;
    Index num_elems; // Number of data elements
    Scalar alpha;
    Scalar beta;
    Scalar eps_check;

    std::vector<T> dst_init;
    std::vector<T> dst_ref;
};

// Reference implementation of the add_scalar operation
template<typename T>
void reference_add_scalar(TestData<T>& data)
{
    using Y = typename T::repr_t;
    if (data.num_elems == 0)
    {
        return;
    }
    const ref_t alpha_r = data.alpha;
    const ref_t beta_r = data.beta;

    for(Index i = 0; i < data.num_elems; ++i)
    {
        ref_t dst_val = static_cast<Y>(data.dst_init[i]);
        ref_t result = alpha_r + beta_r * dst_val;
        data.dst_ref[i] = static_cast<T>(static_cast<Y>(result));
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
void generate_data(TestData<T>& data, Index num_elems, DataGen strategy)
{
    using Y = typename T::repr_t;
    data.num_elems = num_elems;

    data.dst_init.resize(num_elems);

    switch(strategy)
    {
        // Non-random input generation
        case DataGen::PRESET:
            for(Index i = 0; i < num_elems; ++i)
            {
                data.dst_init[i] = Y(2 * i + 1 - num_elems);
            }
            break;
        // Specific random initialization
        case DataGen::RANDOM:
            std::mt19937 gen(42);
            std::uniform_real_distribution<Y> dist(-2.0, 2.0);
            for(Index i = 0; i < num_elems; ++i)
            {
                data.dst_init[i] = dist(gen);
            }
    }
}

// Get test data and reference results
template<typename T>
TestData<T> get_test_data(
    Index num_elems,
    Scalar alpha,
    Scalar beta,
    DataGen strategy
)
{
    TestData<T> data;
    // Generate data by a provided strategy
    generate_data(data, num_elems, strategy);
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
    data.dst_ref = data.dst_init;
    reference_add_scalar(data);
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
    for(Index i = 0; i < data.num_elems; ++i)
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
            "[kernel][add_scalar][cpu][nelems=" +
            std::to_string(data.num_elems) +
            "][alpha=" +
            std::to_string(data.alpha) +
            "][beta=" +
            std::to_string(data.beta) +
            "]"
        )
        {
            cpu<T>(
                data.num_elems,
                data.alpha,
                data.beta,
                &dst_cpu[0]
            );
        };
    }
    else
    {
        cpu<T>(
            data.num_elems,
            data.alpha,
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
    T *dev_dst;
    cudaMalloc(&dev_dst, sizeof(T) * data.num_elems);

    std::vector<T> dst_cuda(data.dst_init);

    cudaMemcpy(dev_dst, &dst_cuda[0], sizeof(T) * data.num_elems,
        cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    if constexpr (run_bench)
    {
        BENCHMARK(
            "[kernel][add_scalar][cuda][nelems=" +
            std::to_string(data.num_elems) +
            "][alpha=" +
            std::to_string(data.alpha) +
            "][beta=" +
            std::to_string(data.beta) +
            "]"
        )
        {
            cuda<T>(
                stream,
                data.num_elems,
                data.alpha,
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
            data.num_elems,
            data.alpha,
            data.beta,
            dev_dst
        );
        cudaStreamSynchronize(stream);

        cudaMemcpy(&dst_cuda[0], dev_dst, sizeof(T) * data.num_elems,
            cudaMemcpyDeviceToHost);

        verify_results(data, dst_cuda);
    }

    cudaFree(dev_dst);
    cudaStreamDestroy(stream);
}
#endif

// Catch2-based tests
TEMPLATE_TEST_CASE(
    "Add Scalar Kernel Verification",
    "[add_scalar]",
    fp64_t,
    fp32_t,
    fp16_t,
    bf16_t
)
{
    using T = TestType;
    const Index num_elems = GENERATE(5, 129);
    const Scalar alpha = GENERATE(-2.0, -1.0, 0.0, 1.0, 2.0);
    const Scalar beta = GENERATE(0.5, 1.0, 2.0);
    const DataGen strategy = GENERATE(DataGen::PRESET, DataGen::RANDOM);

    auto data = get_test_data<T>(
        num_elems,
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
    "Add Scalar Kernel Benchmark",
    "[add_scalar][!benchmark]",
    fp64_t,
    fp32_t,
    fp16_t,
    bf16_t
)
{
    using T = TestType;
    const Index num_elems = GENERATE(512, 1024*1024, 4096*16384);
    const Scalar alpha = GENERATE(1.0);
    const Scalar beta = GENERATE(1.0);
    const DataGen strategy = GENERATE(DataGen::PRESET);

    auto data = get_test_data<T>(
        num_elems,
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