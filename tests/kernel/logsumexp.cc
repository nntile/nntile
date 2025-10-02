/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/logsumexp.cc
 * Logsumexp after computed maxsumexp result of a buffer
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/kernel/logsumexp.hh"

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
using namespace nntile::kernel::logsumexp;

// Type to acquire reference values
using ref_t = double;

// Struct to hold test data and reference results
template<typename T>
struct TestData
{
    using Y = typename T::repr_t;
    Index num_elems; // Number of elements in logsumexp output
    Scalar eps_check;

    std::vector<T> maxsumexp;    // Size: 2*num_elems (interleaved max and sumexp)
    std::vector<T> logsumexp_ref; // Size: num_elems
};

// Reference implementation of the logsumexp operation
template<typename T>
void reference_logsumexp(TestData<T>& data)
{
    using Y = typename T::repr_t;

    for(Index i = 0; i < data.num_elems; ++i)
    {
        ref_t max_val = static_cast<Y>(data.maxsumexp[2*i]);
        ref_t sum_val = static_cast<Y>(data.maxsumexp[2*i+1]);
        ref_t result = max_val + std::log(sum_val);
        data.logsumexp_ref[i] = static_cast<T>(static_cast<Y>(result));
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

    data.maxsumexp.resize(2 * num_elems);
    data.logsumexp_ref.resize(num_elems);

    switch(strategy)
    {
        // Non-random input generation
        case DataGen::PRESET:
            for(Index i = 0; i < num_elems; ++i)
            {
                data.maxsumexp[2*i] = Y(i + 1);      // max value
                data.maxsumexp[2*i+1] = Y(10.0 + i); // sum of exponents
            }
            break;
        // Specific random initialization
        case DataGen::RANDOM:
            std::mt19937 gen(42);
            std::uniform_real_distribution<Y> dist_max(-2.0, 2.0);
            std::uniform_real_distribution<Y> dist_sum(1.0, 20.0);
            for(Index i = 0; i < num_elems; ++i)
            {
                data.maxsumexp[2*i] = dist_max(gen);     // max value
                data.maxsumexp[2*i+1] = dist_sum(gen);   // sum of exponents
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
        data.eps_check = 1e-5;
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
    reference_logsumexp(data);
    return data;
}

// Helper function to verify results
template<typename T>
void verify_results(
    const TestData<T>& data,
    const std::vector<T>& logsumexp_out
)
{
    using Y = typename T::repr_t;
    for(Index i = 0; i < data.num_elems; ++i)
    {
        Y logsumexp_ref = static_cast<Y>(data.logsumexp_ref[i]);
        auto logsumexp_approx = Approx(logsumexp_ref).epsilon(data.eps_check);
        REQUIRE(static_cast<Y>(logsumexp_out[i]) == logsumexp_approx);
    }
}

// Helper function to run CPU test and verify results
template<typename T, bool run_bench>
void run_cpu_test(TestData<T>& data)
{
    std::vector<T> logsumexp_cpu(data.num_elems);

    if constexpr (run_bench)
    {
        BENCHMARK(
            "[kernel][logsumexp][cpu][nelems=" +
            std::to_string(data.num_elems) +
            "]"
        )
        {
            cpu<T>(
                data.num_elems,
                &data.maxsumexp[0],
                &logsumexp_cpu[0]
            );
        };
    }
    else
    {
        cpu<T>(
            data.num_elems,
            &data.maxsumexp[0],
            &logsumexp_cpu[0]
        );
        verify_results(data, logsumexp_cpu);
    }
}

#ifdef NNTILE_USE_CUDA

// Helper function to run CUDA test and verify results
template<typename T, bool run_bench>
void run_cuda_test(TestData<T>& data)
{
    T *dev_maxsumexp, *dev_logsumexp;
    CUDA_CHECK(cudaMalloc(&dev_maxsumexp, sizeof(T) * 2 * data.num_elems),
               "cudaMalloc dev_maxsumexp");
    CUDA_CHECK(cudaMalloc(&dev_logsumexp, sizeof(T) * data.num_elems),
               "cudaMalloc dev_logsumexp");

    std::vector<T> logsumexp_cuda(data.num_elems);

    CUDA_CHECK(cudaMemcpy(dev_maxsumexp, &data.maxsumexp[0],
                          sizeof(T) * 2 * data.num_elems,
                          cudaMemcpyHostToDevice), "cudaMemcpy dev_maxsumexp");

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream), "cudaStreamCreate");

    if constexpr (run_bench)
    {
        BENCHMARK(
            "[kernel][logsumexp][cuda][nelems=" +
            std::to_string(data.num_elems) +
            "]"
        )
        {
            cuda<T>(
                stream,
                data.num_elems,
                dev_maxsumexp,
                dev_logsumexp
            );
            cudaStreamSynchronize(stream);
        };
    }
    else
    {
        cuda<T>(
            stream,
            data.num_elems,
            dev_maxsumexp,
            dev_logsumexp
        );
        CUDA_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

        CUDA_CHECK(cudaMemcpy(&logsumexp_cuda[0], dev_logsumexp,
                              sizeof(T) * data.num_elems,
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy logsumexp_cuda");

        verify_results(data, logsumexp_cuda);
    }

    CUDA_CHECK(cudaFree(dev_maxsumexp), "cudaFree dev_maxsumexp");
    CUDA_CHECK(cudaFree(dev_logsumexp), "cudaFree dev_logsumexp");
    CUDA_CHECK(cudaStreamDestroy(stream), "cudaStreamDestroy");
}
#endif

// Catch2-based tests
TEMPLATE_TEST_CASE(
    "Logsumexp Kernel Verification",
    "[logsumexp]",
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
    "Logsumexp Kernel Benchmark",
    "[logsumexp][!benchmark]",
    fp64_t,
    fp32_t,
    fp16_t,
    bf16_t
)
{
    using T = TestType;
    const Index num_elems = GENERATE(512, 1024*1024);
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
