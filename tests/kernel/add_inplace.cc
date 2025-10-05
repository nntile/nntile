/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/add_inplace.cc
 * Per-element addition of tensors
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/kernel/add_inplace.hh"

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
using namespace nntile::kernel::add_inplace;

// Type to acquire reference values
using ref_t = double;

// Struct to hold test data and reference results
template<typename T>
struct TestData
{
    using Y = typename T::repr_t;
    Index nelems; // Number of elements
    Scalar alpha, beta; // Scalar factors

    Y eps_check;

    std::vector<T> src_init;
    std::vector<T> dst_init;

    std::vector<T> dst_ref;
};

// Reference implementation of the add inplace operation
template<typename T>
void reference_add_inplace(TestData<T>& data)
{
    using Y = typename T::repr_t;

    data.dst_ref = data.dst_init; // Copy initial destination

    for(Index i = 0; i < data.nelems; ++i)
    {
        const ref_t src_val = static_cast<Y>(data.src_init[i]);
        ref_t dst_val = static_cast<Y>(data.dst_ref[i]);
        dst_val = data.beta * dst_val + src_val * data.alpha;
        data.dst_ref[i] = static_cast<Y>(dst_val);
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

    data.src_init.resize(data.nelems);
    data.dst_init.resize(data.nelems);
    data.dst_ref.resize(data.nelems);

    switch(strategy)
    {
        // Non-random input generation
        case DataGen::PRESET:
            for(Index i = 0; i < data.nelems; ++i)
            {
                Index tmp_i = 2 * i + 1 - data.nelems;
                data.src_init[i] = static_cast<Y>(tmp_i);
                Index tmp_i2 = 2 * data.nelems - i;
                data.dst_init[i] = static_cast<Y>(tmp_i2);
            }
            break;
        // Specific random initialization
        case DataGen::RANDOM:
            std::mt19937 gen(42);
            std::uniform_real_distribution<Y> dist(-2.0, 2.0);
            for(Index i = 0; i < data.nelems; ++i)
            {
                data.src_init[i] = dist(gen);
                data.dst_init[i] = dist(gen);
            }
            break;
    }
}

// Get test data and reference results
template<typename T>
TestData<T> get_test_data(
    Index nelems,
    Scalar alpha,
    Scalar beta,
    DataGen strategy
)
{
    using Y = typename T::repr_t;
    TestData<T> data;
    data.nelems = nelems;
    data.alpha = alpha;
    data.beta = beta;

    // Generate data by a provided strategy
    generate_data(data, strategy);

    // Set accuracy threshold for each precision
    if (std::is_same_v<T, bf16_t>)
    {
        data.eps_check = Y{1e-1};
    }
    else if (std::is_same_v<T, fp16_t>)
    {
        data.eps_check = Y{1e-2};
    }
    else if (std::is_same_v<T, fp32_t>)
    {
        data.eps_check = Y{1e-6};
    }
    else if (std::is_same_v<T, fp64_t>)
    {
        data.eps_check = Y{1e-12};
    }
    else
    {
        throw std::runtime_error("Unsupported data type");
    }

    // Compute reference outputs
    reference_add_inplace(data);
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

    // Check that source data was not modified
    for(Index i = 0; i < data.nelems; ++i)
    {
        REQUIRE(static_cast<Y>(src[i]) == static_cast<Y>(data.src_init[i]));
    }

    for(Index i = 0; i < data.nelems; ++i)
    {
        REQUIRE_THAT(
            static_cast<Y>(dst[i]),
            WithinRel(static_cast<Y>(data.dst_ref[i]), data.eps_check)
        );
    }
}

// Helper function to run CPU test and verify results
template<typename T, bool run_bench>
void run_cpu_test(TestData<T>& data)
{
    std::vector<T> dst_cpu(data.dst_init);
    std::vector<T> src_cpu(data.src_init);

    if constexpr (run_bench)
    {
        BENCHMARK(
            "[kernel][add_inplace][cpu][nelems=" +
            std::to_string(data.nelems) +
            "][alpha=" + std::to_string(data.alpha) +
            "][beta=" + std::to_string(data.beta) + "]"
        )
        {
            cpu<T>(
                data.nelems,
                data.alpha,
                &src_cpu[0],
                data.beta,
                &dst_cpu[0]
            );
        };
    }
    else
    {
        cpu<T>(data.nelems, data.alpha, &src_cpu[0], data.beta, &dst_cpu[0]);
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
        cudaMalloc(&dev_src, sizeof(T) * data.nelems),
        "cudaMalloc dev_src"
    );
    CUDA_CHECK(
        cudaMalloc(&dev_dst, sizeof(T) * data.nelems),
        "cudaMalloc dev_dst"
    );

    std::vector<T> dst_cuda(data.dst_init);
    std::vector<T> src_cuda(data.src_init);

    CUDA_CHECK(
        cudaMemcpy(
            dev_src,
            &src_cuda[0],
            sizeof(T) * data.nelems,
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy dev_src"
    );
    CUDA_CHECK(
        cudaMemcpy(
            dev_dst,
            &dst_cuda[0],
            sizeof(T) * data.nelems,
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy dev_dst"
    );

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream), "cudaStreamCreate");

    if constexpr (run_bench)
    {
        BENCHMARK(
            "[kernel][add_inplace][cuda][nelems=" +
            std::to_string(data.nelems) +
            "][alpha=" + std::to_string(data.alpha) +
            "][beta=" + std::to_string(data.beta) + "]"
        )
        {
            cuda<T>(
                stream,
                data.nelems,
                data.alpha,
                dev_src,
                data.beta,
                dev_dst
            );
            cudaStreamSynchronize(stream);
        };
    }
    else
    {
        cuda<T>(stream, data.nelems, data.alpha, dev_src, data.beta, dev_dst);
        CUDA_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

        CUDA_CHECK(
            cudaMemcpy(
                &dst_cuda[0],
                dev_dst,
                sizeof(T) * data.nelems,
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy dst_cuda"
        );
        CUDA_CHECK(
            cudaMemcpy(
                &src_cuda[0],
                dev_src,
                sizeof(T) * data.nelems,
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy src_cuda"
        );

        verify_results(data, src_cuda, dst_cuda);
    }

    CUDA_CHECK(cudaFree(dev_src), "cudaFree dev_src");
    CUDA_CHECK(cudaFree(dev_dst), "cudaFree dev_dst");
    CUDA_CHECK(cudaStreamDestroy(stream), "cudaStreamDestroy");
}
#endif

// Catch2-based tests
TEMPLATE_TEST_CASE(
    "Add Inplace Kernel Verification",
    "[add_inplace]",
    fp64_t,
    fp32_t,
    fp16_t,
    bf16_t
)
{
    using T = TestType;
    const Index nelems = GENERATE(0, 3, 999);
    const Scalar alpha = GENERATE(0.5, 1.0, 2.0);
    const Scalar beta = GENERATE(0.0, 0.5, 1.0);
    const DataGen strategy = GENERATE(DataGen::PRESET, DataGen::RANDOM);

    auto data = get_test_data<T>(nelems, alpha, beta, strategy);

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
    "Add Inplace Kernel Benchmark",
    "[add_inplace][!benchmark]",
    fp64_t,
    fp32_t,
    fp16_t,
    bf16_t
)
{
    using T = TestType;
    const Index nelems = GENERATE(1000, 10000, 100000);
    const Scalar alpha = GENERATE(1.0);
    const Scalar beta = GENERATE(1.0);
    const DataGen strategy = GENERATE(DataGen::PRESET);

    auto data = get_test_data<T>(nelems, alpha, beta, strategy);

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
