/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/gelutanh_inplace.cc
 * Approximate GeLU operation on a buffer
 *
 * @version 1.1.0
 */

// Corresponding header
#include "nntile/kernel/gelutanh_inplace.hh"

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
using namespace nntile::kernel::gelutanh_inplace;

// Type to acquire reference values
using ref_t = double;

// Struct to hold test data and reference results
template<typename T>
struct TestData
{
    using Y = typename T::repr_t;
    Index num_elems; // Number of elements

    Y eps_check;

    std::vector<T> data_init; // Initial data
    std::vector<T> data_ref;  // Reference result
};

// Reference implementation of the approximate GeLU operation
template<typename T>
void reference_gelutanh_inplace(TestData<T>& data)
{
    using Y = typename T::repr_t;
    if (data.num_elems == 0)
    {
        return;
    }
    constexpr Y pi{3.141592653589793238462643383279502884L};
    const Y sqrt_pi = std::sqrt(pi), sqrt_2 = std::sqrt(Y{2.0}),
        f2 = sqrt_2/sqrt_pi, f3 = -Y{2.0}*f2, f4 = f3*Y{0.044715};

    for(Index i = 0; i < data.num_elems; ++i)
    {
        Y x = static_cast<Y>(data.data_init[i]);
        Y y = x * (f3 + f4*x*x);
        Y val_ref = x / (Y{1.0} + std::exp(y));
        data.data_ref[i] = static_cast<T>(val_ref);
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

    data.data_init.resize(num_elems);
    data.data_ref.resize(num_elems);

    switch(strategy)
    {
        // Non-random input generation
        case DataGen::PRESET:
            for(Index i = 0; i < num_elems; ++i)
            {
                data.data_init[i] = Y(2 * i + 1 - num_elems) / Y{1000};
            }
            break;
        // Specific random initialization
        case DataGen::RANDOM:
            std::mt19937 gen(42);
            std::uniform_real_distribution<Y> dist(-3.0, 3.0);
            for(Index i = 0; i < num_elems; ++i)
            {
                data.data_init[i] = dist(gen);
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
    reference_gelutanh_inplace(data);
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

    // Check that output matches reference (this is an in-place operation)
    for(Index i = 0; i < data.num_elems; ++i)
    {
        const Y val_ref = static_cast<Y>(data.data_ref[i]);
        const Y val_out = static_cast<Y>(data_out[i]);

        // Obtain range of correct values for floating point comparisons
        Y val_ref_min, val_ref_max;
        if(val_ref < 0)
        {
            val_ref_min = val_ref * (Y{1}+data.eps_check) - data.eps_check;
            val_ref_max = val_ref * (Y{1}-data.eps_check) + data.eps_check;
        }
        else
        {
            val_ref_min = val_ref * (Y{1}-data.eps_check) - data.eps_check;
            val_ref_max = val_ref * (Y{1}+data.eps_check) + data.eps_check;
        }

        // NaN-aware comparisons
        REQUIRE((val_out >= val_ref_min && val_out <= val_ref_max));
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
            "[kernel][gelutanh_inplace][cpu][nelems=" +
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

    std::vector<T> data_cuda(data.data_init);

    CUDA_CHECK(cudaMemcpy(dev_data, &data_cuda[0], sizeof(T) * data.num_elems,
                          cudaMemcpyHostToDevice), "cudaMemcpy dev_data");

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream), "cudaStreamCreate");

    if constexpr (run_bench)
    {
        BENCHMARK(
            "[kernel][gelutanh_inplace][cuda][nelems=" +
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

// Catch2-based tests
TEMPLATE_TEST_CASE(
    "GeLUTanh Inplace Kernel Verification",
    "[gelutanh_inplace]",
    fp64_t,
    fp32_t,
    bf16_t
)
{
    using T = TestType;
    const Index num_elems = GENERATE(0, 1, 100, 1000, 10000);
    const DataGen strategy = GENERATE(DataGen::PRESET, DataGen::RANDOM);

    auto data = get_test_data<T>(
        num_elems,
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
    "GeLUTanh Inplace Kernel Benchmark",
    "[gelutanh_inplace][!benchmark]",
    fp64_t,
    fp32_t,
    bf16_t
)
{
    using T = TestType;
    const Index num_elems = GENERATE(1024, 1024*1024, 4096*4096);
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