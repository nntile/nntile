/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/add.cc
 * Per-element addition of tensors
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/kernel/add.hh"

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
using namespace nntile::kernel::add;

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

    Y eps_check;

    std::vector<T> src1;
    std::vector<T> src2;
    std::vector<T> dst_ref;
};

// Reference implementation of the add operation
template<typename T>
void reference_add(TestData<T>& data)
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
        ref_t src1_val = static_cast<Y>(data.src1[i]);
        ref_t src2_val = static_cast<Y>(data.src2[i]);
        ref_t result = alpha_r * src1_val + beta_r * src2_val;
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

    data.src1.resize(num_elems);
    data.src2.resize(num_elems);
    data.dst_ref.resize(num_elems);

    switch(strategy)
    {
        // Non-random input generation
        case DataGen::PRESET:
            for(Index i = 0; i < num_elems; ++i)
            {
                data.src1[i] = Y(2 * i + 1 - num_elems);
                data.src2[i] = Y(2 * num_elems - i);
            }
            break;
        // Specific random initialization
        case DataGen::RANDOM:
            std::mt19937 gen(42);
            std::uniform_real_distribution<Y> dist(1.0, 2.0);
            for(Index i = 0; i < num_elems; ++i)
            {
                data.src1[i] = dist(gen);
                data.src2[i] = dist(gen);
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
    reference_add(data);
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
    std::vector<T> dst_cpu(data.num_elems);

    if constexpr (run_bench)
    {
        BENCHMARK(
            "[kernel][add][cpu][nelems=" +
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
            data.num_elems,
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
    CUDA_CHECK(cudaMalloc(&dev_src1, sizeof(T) * data.num_elems),
               "cudaMalloc dev_src1");
    CUDA_CHECK(cudaMalloc(&dev_src2, sizeof(T) * data.num_elems),
               "cudaMalloc dev_src2");
    CUDA_CHECK(cudaMalloc(&dev_dst, sizeof(T) * data.num_elems),
               "cudaMalloc dev_dst");

    std::vector<T> dst_cuda(data.num_elems);

    CUDA_CHECK(cudaMemcpy(dev_src1, &data.src1[0], sizeof(T) * data.num_elems,
                          cudaMemcpyHostToDevice), "cudaMemcpy dev_src1");
    CUDA_CHECK(cudaMemcpy(dev_src2, &data.src2[0], sizeof(T) * data.num_elems,
                          cudaMemcpyHostToDevice), "cudaMemcpy dev_src2");
    CUDA_CHECK(cudaMemcpy(dev_dst, &dst_cuda[0], sizeof(T) * data.num_elems,
                          cudaMemcpyHostToDevice), "cudaMemcpy dev_dst");

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream), "cudaStreamCreate");

    if constexpr (run_bench)
    {
        BENCHMARK(
            "[kernel][add][cuda][nelems=" +
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
            data.num_elems,
            data.alpha,
            dev_src1,
            data.beta,
            dev_src2,
            dev_dst
        );
        CUDA_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

        CUDA_CHECK(cudaMemcpy(&dst_cuda[0], dev_dst, sizeof(T) * data.num_elems,
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
    "Add Kernel Verification",
    "[add]",
    fp64_t,
    fp32_t,
    fp16_t,
    bf16_t
)
{
    using T = TestType;
    const Index num_elems = GENERATE(5, 129);
    const Scalar alpha = GENERATE(0.1, 1.0, 2.5);
    const Scalar beta = GENERATE(0.5, 1.5, 3.0);
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
    "Add Kernel Benchmark",
    "[add][!benchmark]",
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
