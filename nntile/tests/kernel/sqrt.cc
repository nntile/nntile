/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/sqrt.cc
 * Per-element square root operation
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/kernel/sqrt.hh"

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
using namespace nntile::kernel::sqrt;

// Type to acquire reference values
using ref_t = double;

// Struct to hold test data and reference results
template<typename T>
struct TestData
{
    using Y = typename T::repr_t;
    Index num_elems; // Number of data elements

    Y eps_check;

    std::vector<T> src_init;
    std::vector<T> dst_init;

    std::vector<T> dst_ref;
};

// Reference implementation of the sqrt operation
template<typename T>
void reference_sqrt(TestData<T>& data)
{
    using Y = typename T::repr_t;
    if (data.num_elems == 0)
    {
        return;
    }

    for(Index i = 0; i < data.num_elems; ++i)
    {
        ref_t x = static_cast<Y>(data.src_init[i]);
        data.dst_ref[i] = static_cast<Y>(std::sqrt(x));
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

    data.src_init.resize(num_elems);
    data.dst_ref.resize(num_elems);
    data.dst_init.resize(num_elems);

    switch(strategy)
    {
        // Non-random input generation
        case DataGen::PRESET:
            for(Index i = 0; i < num_elems; ++i)
            {
                Y src_val = i + 1;
                Y dst_val = i % 3 - 1;
                data.src_init[i] = src_val;
                data.dst_init[i] = dst_val;
            }
            break;
        // Specific random initialization
        case DataGen::RANDOM:
            std::mt19937 gen(42);
            std::uniform_real_distribution<Y> dist(0.1, 4.0);
            for(Index i = 0; i < num_elems; ++i)
            {
                data.src_init[i] = dist(gen);
                data.dst_init[i] = dist(gen);
            }
    }
}

// Get test input data (reference computation is done separately)
template<typename T>
TestData<T> get_test_input_data(
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
        data.eps_check = 1e-6;
    }
    else if (std::is_same_v<T, fp64_t>)
    {
        data.eps_check = 1e-12;
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
    const std::vector<T>& src,
    const std::vector<T>& dst
)
{
    using Y = typename T::repr_t;
    // Check that src was not changed during kernel execution
    for(Index i = 0; i < data.num_elems; ++i)
    {
        REQUIRE(static_cast<Y>(src[i]) == static_cast<Y>(data.src_init[i]));
    }

    // Check that dst matches reference
    for(Index i = 0; i < data.num_elems; ++i)
    {
        Y dst_ref = static_cast<Y>(data.dst_ref[i]);
        REQUIRE_THAT(
            static_cast<Y>(dst[i]),
            WithinRel(dst_ref, data.eps_check)
        );
    }
}

// Helper function to run CPU test and verify results
template<typename T, bool run_bench>
void run_cpu_test(TestData<T>& data)
{
    std::vector<T> src_cpu(data.src_init);
    std::vector<T> dst_cpu(data.dst_init);

    if constexpr (run_bench)
    {
        BENCHMARK(
            "[kernel][sqrt][cpu][nelems=" +
            std::to_string(data.num_elems) +
            "]"
        )
        {
            cpu<T>(
                data.num_elems,
                &src_cpu[0],
                &dst_cpu[0]
            );
        };
    }
    else
    {
        cpu<T>(
            data.num_elems,
            &src_cpu[0],
            &dst_cpu[0]
        );
        verify_results(data, src_cpu, dst_cpu);
    }
}

#ifdef NNTILE_USE_CUDA
// Helper function to run CUDA test and verify results
template<typename T, bool run_bench>
void run_cuda_test(TestData<T>& data)
{
    T *dev_src, *dev_dst;
    CUDA_CHECK(
        cudaMalloc(
            &dev_src,
            sizeof(T) * data.num_elems
        ),
        "cudaMalloc dev_src"
    );
    CUDA_CHECK(
        cudaMalloc(
            &dev_dst,
            sizeof(T) * data.num_elems
        ),
        "cudaMalloc dev_dst"
    );

    std::vector<T> src_cuda(data.src_init);
    std::vector<T> dst_cuda(data.dst_init);

    CUDA_CHECK(
        cudaMemcpy(
            dev_src,
            &src_cuda[0],
            sizeof(T) * data.num_elems,
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy dev_src"
    );
    CUDA_CHECK(
        cudaMemcpy(
            dev_dst,
            &dst_cuda[0],
            sizeof(T) * data.num_elems,
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy dev_dst"
    );

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream), "cudaStreamCreate");

    if constexpr (run_bench)
    {
        BENCHMARK(
            "[kernel][sqrt][cuda][nelems=" +
            std::to_string(data.num_elems) +
            "]"
        )
        {
            cuda<T>(
                stream,
                data.num_elems,
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
            data.num_elems,
            dev_src,
            dev_dst
        );
        CUDA_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

        CUDA_CHECK(
            cudaMemcpy(
                &dst_cuda[0],
                dev_dst,
                sizeof(T) * data.num_elems,
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy dst_cuda"
        );

        verify_results(data, src_cuda, dst_cuda);
    }

    CUDA_CHECK(cudaFree(dev_src), "cudaFree dev_src");
    CUDA_CHECK(cudaFree(dev_dst), "cudaFree dev_dst");
    CUDA_CHECK(cudaStreamDestroy(stream), "cudaStreamDestroy");
}
#endif // NNTILE_USE_CUDA

// Catch2-based tests
TEMPLATE_TEST_CASE(
    "Sqrt Kernel Verification",
    "[sqrt]",
    fp64_t,
    fp32_t,
    fp16_t,
    bf16_t
)
{
    using T = TestType;
    const Index num_elems = GENERATE(5, 129);
    const DataGen strategy = GENERATE(DataGen::PRESET, DataGen::RANDOM);

    auto data = get_test_input_data<T>(
        num_elems,
        strategy
    );

    // Compute reference outputs for verification
    reference_sqrt(data);

    SECTION("cpu")
    {
        run_cpu_test<T, false>(data);
    }

#ifdef NNTILE_USE_CUDA
    SECTION("cuda")
    {
        run_cuda_test<T, false>(data);
    }
#endif // NNTILE_USE_CUDA
}

// Catch2-based benchmarks
TEMPLATE_TEST_CASE(
    "Sqrt Kernel Benchmark",
    "[sqrt][!benchmark]",
    fp64_t,
    fp32_t,
    fp16_t,
    bf16_t
)
{
    using T = TestType;
    const Index num_elems = GENERATE(512, 1024*1024);
    const DataGen strategy = GENERATE(DataGen::PRESET);

    auto data = get_test_input_data<T>(
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
#endif // NNTILE_USE_CUDA
}
