/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/gelu_backward.cc
 * Backward GeLU operation
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/kernel/gelu_backward.hh"

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
using namespace nntile::kernel::gelu_backward;

// Type to acquire reference values
using ref_t = double;

// Struct to hold test data and reference results
template<typename T>
struct TestData
{
    using Y = typename T::repr_t;
    Index nelems; // Number of elements
    Scalar eps_check;

    std::vector<T> x;
    std::vector<T> dy;
    std::vector<T> dx_ref;
};

// Reference implementation of the backward GeLU operation
template<typename T>
void reference_gelu_backward(TestData<T>& data)
{
    using Y = typename T::repr_t;
    if (data.nelems == 0)
    {
        return;
    }
    constexpr Y pi{3.141592653589793238462643383279502884L},
        one{1.0}, mone{-1.0}, pt5{0.5};
    const Y f1 = mone / std::sqrt(Y{2.0}), f2 = one / std::sqrt(2*pi);

    for(Index i = 0; i < data.nelems; ++i)
    {
        Y exp_x = std::exp(-pt5 * Y{data.x[i]} * Y{data.x[i]});
        Y y = std::erfc(f1 * Y{data.x[i]});
        data.dx_ref[i] = T{(Y{data.x[i]}*f2*exp_x + pt5*y) * Y{data.dy[i]}};
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
                data.x[i] = Y(2 * i + 1 - data.nelems);
                data.dy[i] = Y(i + 1);
            }
            break;
        // Specific random initialization
        case DataGen::RANDOM:
            std::mt19937 gen(42);
            std::uniform_real_distribution<Y> dist(-2.0, 2.0);
            for(Index i = 0; i < data.nelems; ++i)
            {
                data.x[i] = dist(gen);
                data.dy[i] = dist(gen) * 0.1; // Smaller gradients for stability
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
    data.x.resize(nelems);
    data.dy.resize(nelems);
    data.dx_ref.resize(nelems);
    generate_data(data, strategy);

    // Set accuracy threshold for each precision
    if (std::is_same_v<T, bf16_t>)
    {
        data.eps_check = 1e-1;
    }
    else if (std::is_same_v<T, fp16_t>)
    {
        data.eps_check = 5e-2;  // Increased tolerance for fp16 precision
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
    reference_gelu_backward(data);
    return data;
}

// Helper function to verify results
template<typename T>
void verify_results(
    const TestData<T>& data,
    const std::vector<T>& dx_out
)
{
    using Y = typename T::repr_t;
    for(Index i = 0; i < data.nelems; ++i)
    {
        Y dx_ref = static_cast<Y>(data.dx_ref[i]);
        auto dx_approx = Approx(dx_ref).epsilon(data.eps_check);
        REQUIRE(static_cast<Y>(dx_out[i]) == dx_approx);
    }
}

// Helper function to run CPU test and verify results
template<typename T, bool run_bench>
void run_cpu_test(TestData<T>& data)
{
    std::vector<T> dx_cpu(data.nelems, T{0}); // Initialize with zeros

    if constexpr (run_bench)
    {
        BENCHMARK(
            "[kernel][gelu_backward][cpu][nelems=" +
            std::to_string(data.nelems) +
            "]"
        )
        {
            cpu<T>(
                data.nelems,
                &data.x[0],
                &data.dy[0],
                &dx_cpu[0]
            );
        };
    }
    else
    {
        cpu<T>(
            data.nelems,
            &data.x[0],
            &data.dy[0],
            &dx_cpu[0]
        );
        verify_results(data, dx_cpu);
    }
}

#ifdef NNTILE_USE_CUDA

// Helper function to run CUDA test and verify results
template<typename T, bool run_bench>
void run_cuda_test(TestData<T>& data)
{
    T *dev_x, *dev_dy, *dev_dx;
    CUDA_CHECK(cudaMalloc(&dev_x, sizeof(T) * data.nelems),
               "cudaMalloc dev_x");
    CUDA_CHECK(cudaMalloc(&dev_dy, sizeof(T) * data.nelems),
               "cudaMalloc dev_dy");
    CUDA_CHECK(cudaMalloc(&dev_dx, sizeof(T) * data.nelems),
               "cudaMalloc dev_dx");

    std::vector<T> dx_cuda(data.dx_ref); // Initialize with zeros

    CUDA_CHECK(cudaMemcpy(dev_x, &data.x[0], sizeof(T) * data.nelems,
                          cudaMemcpyHostToDevice), "cudaMemcpy dev_x");
    CUDA_CHECK(cudaMemcpy(dev_dy, &data.dy[0], sizeof(T) * data.nelems,
                          cudaMemcpyHostToDevice), "cudaMemcpy dev_dy");
    CUDA_CHECK(cudaMemcpy(dev_dx, &dx_cuda[0], sizeof(T) * data.nelems,
                          cudaMemcpyHostToDevice), "cudaMemcpy dev_dx");

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream), "cudaStreamCreate");

    if constexpr (run_bench)
    {
        BENCHMARK(
            "[kernel][gelu_backward][cuda][nelems=" +
            std::to_string(data.nelems) +
            "]"
        )
        {
            cuda<T>(
                stream,
                data.nelems,
                dev_x,
                dev_dy,
                dev_dx
            );
            cudaStreamSynchronize(stream);
        };
    }
    else
    {
        cuda<T>(
            stream,
            data.nelems,
            dev_x,
            dev_dy,
            dev_dx
        );
        CUDA_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

        CUDA_CHECK(cudaMemcpy(&dx_cuda[0], dev_dx, sizeof(T) * data.nelems,
                              cudaMemcpyDeviceToHost), "cudaMemcpy dx_cuda");

        verify_results(data, dx_cuda);
    }

    CUDA_CHECK(cudaFree(dev_x), "cudaFree dev_x");
    CUDA_CHECK(cudaFree(dev_dy), "cudaFree dev_dy");
    CUDA_CHECK(cudaFree(dev_dx), "cudaFree dev_dx");
    CUDA_CHECK(cudaStreamDestroy(stream), "cudaStreamDestroy");
}
#endif

// Catch2-based tests
TEMPLATE_TEST_CASE(
    "GeLU Backward Kernel Verification",
    "[gelu_backward]",
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
    "GeLU Backward Kernel Benchmark",
    "[gelu_backward][!benchmark]",
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
