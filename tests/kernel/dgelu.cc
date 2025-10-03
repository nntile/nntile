/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/dgelu.cc
 * Derivative of GeLU operation on a buffer
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/kernel/dgelu.hh"
#include "nntile/kernel/gelu.hh"

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
using namespace nntile::kernel::dgelu;

// Type to acquire reference values
using ref_t = double;

// Struct to hold test data and reference results
template<typename T>
struct TestData
{
    using Y = typename T::repr_t;
    Index num_elems; // Number of data elements

    Y eps_check;

    std::vector<T> data;
    std::vector<T> data_ref;
};

// Reference implementation of the dgelu operation
template<typename T>
void reference_dgelu(TestData<T>& data)
{
    using Y = typename T::repr_t;
    if (data.num_elems == 0)
    {
        return;
    }
    constexpr Y pi = 3.141592653589793238462643383279502884L;

    for(Index i = 0; i < data.num_elems; ++i)
    {
        Y x = static_cast<Y>(data.data[i]);
        Y val_ref = 0.5 * std::erfc(-x/std::sqrt(Y(2)));
        val_ref += x / std::sqrt(2*pi) * std::exp(-0.5*x*x);
        data.data_ref[i] = static_cast<T>(static_cast<Y>(val_ref));
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

    data.data.resize(num_elems);
    data.data_ref.resize(num_elems);

    switch(strategy)
    {
        // Non-random input generation
        case DataGen::PRESET:
            for(Index i = 0; i < num_elems; ++i)
            {
                data.data[i] = Y(2 * i + 1 - num_elems) / Y{1000};
            }
            break;
        // Specific random initialization
        case DataGen::RANDOM:
            std::mt19937 gen(42);
            std::uniform_real_distribution<Y> dist(-2.0, 2.0);
            for(Index i = 0; i < num_elems; ++i)
            {
                data.data[i] = dist(gen);
            }
    }
}

// Get test data and reference results
template<typename T>
TestData<T> get_test_data(
    Index num_elems,
    DataGen strategy
)
{
    TestData<T> data;
    // Generate data by a provided strategy
    generate_data(data, num_elems, strategy);
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
    reference_dgelu(data);
    return data;
}

// Helper function to verify results
template<typename T>
void verify_results(
    const TestData<T>& data,
    const std::vector<T>& data_out
)
{
    using Y = typename T::repr_t;
    for(Index i = 0; i < data.num_elems; ++i)
    {
        Y data_ref = static_cast<Y>(data.data_ref[i]);
        // Obtain range of correct values for numerical stability
        Y eps = 2 * T::epsilon;
        Y val_ref_min, val_ref_max;
        if(data_ref < 0)
        {
            val_ref_min = data_ref * (Y{1}+eps) - eps;
            val_ref_max = data_ref * (Y{1}-eps) + eps;
        }
        else
        {
            val_ref_min = data_ref * (Y{1}-eps) - eps;
            val_ref_max = data_ref * (Y{1}+eps) + eps;
        }
        REQUIRE( (static_cast<Y>(data_out[i]) >= val_ref_min &&
                  static_cast<Y>(data_out[i]) <= val_ref_max) );
    }
}

// Helper function to run CPU test and verify results
template<typename T, bool run_bench>
void run_cpu_test(TestData<T>& data)
{
    std::vector<T> data_cpu(data.data);

    if constexpr (run_bench)
    {
        BENCHMARK(
            "[kernel][dgelu][cpu][nelems=" +
            std::to_string(data.num_elems) +
            "]"
        )
        {
            cpu<T>(
                data.num_elems,
                &data_cpu[0]
            );
        };
    }
    else
    {
        cpu<T>(
            data.num_elems,
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
    CUDA_CHECK(cudaMalloc(&dev_data, sizeof(T) * data.num_elems),
               "cudaMalloc dev_data");

    std::vector<T> data_cuda(data.data);

    CUDA_CHECK(cudaMemcpy(dev_data, &data.data[0], sizeof(T) * data.num_elems,
                          cudaMemcpyHostToDevice), "cudaMemcpy dev_data");

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream), "cudaStreamCreate");

    if constexpr (run_bench)
    {
        BENCHMARK(
            "[kernel][dgelu][cuda][nelems=" +
            std::to_string(data.num_elems) +
            "]"
        )
        {
            cuda<T>(
                stream,
                data.num_elems,
                dev_data
            );
            cudaStreamSynchronize(stream);
        };
    }
    else
    {
        cuda<T>(
            stream,
            data.num_elems,
            dev_data
        );
        CUDA_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

        CUDA_CHECK(cudaMemcpy(&data_cuda[0], dev_data, sizeof(T) * data.num_elems,
                              cudaMemcpyDeviceToHost), "cudaMemcpy data_cuda");

        verify_results(data, data_cuda);
    }

    CUDA_CHECK(cudaFree(dev_data), "cudaFree dev_data");
    CUDA_CHECK(cudaStreamDestroy(stream), "cudaStreamDestroy");
}
#endif

// Helper function to verify that dgelu is the derivative of gelu numerically
template<typename T>
void verify_derivative(TestData<T>& data)
{
    using Y = typename T::repr_t;
    constexpr Y h = 1e-3, inv_h = 1/h;

    std::vector<T> data2(data.data), data3(data.data);

    for(Index i = 0; i < data.num_elems; ++i)
    {
        data2[i] = static_cast<T>(static_cast<Y>(data2[i]) + h/2);
        data3[i] = static_cast<T>(static_cast<Y>(data3[i]) - h/2);
    }

    gelu::cpu<T>(data.num_elems, &data2[0]);
    gelu::cpu<T>(data.num_elems, &data3[0]);

    for(Index i = 0; i < data.num_elems; ++i)
    {
        Y num_x = inv_h * (static_cast<Y>(data2[i]) - static_cast<Y>(data3[i]));
        Y diff = std::abs(num_x - static_cast<Y>(data.data[i]));
        Y abs = std::abs(static_cast<Y>(data.data[i]));
        Y threshold = abs * 5e-2;
        // NaN-aware comparisons
        REQUIRE( (diff <= threshold || (diff > threshold && abs < 1e-4)) );
    }
}

// Catch2-based tests
TEMPLATE_TEST_CASE(
    "DGelu Kernel Verification",
    "[dgelu]",
    fp64_t,
    fp32_t,
    fp16_t,
    bf16_t
)
{
    using T = TestType;
    const Index num_elems = GENERATE(5, 129);
    const DataGen strategy = GENERATE(DataGen::PRESET, DataGen::RANDOM);

    auto data = get_test_data<T>(
        num_elems,
        strategy
    );

    SECTION("cpu")
    {
        run_cpu_test<T, false>(data);
        verify_derivative(data);
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
    "DGelu Kernel Benchmark",
    "[dgelu][!benchmark]",
    fp64_t,
    fp32_t,
    fp16_t,
    bf16_t
)
{
    using T = TestType;
    const Index num_elems = GENERATE(512, 1024*1024, 4096*16384);
    const DataGen strategy = GENERATE(DataGen::PRESET);

    auto data = get_test_data<T>(
        num_elems,
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
