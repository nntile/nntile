/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/accumulate_maxsumexp.cc
 * Placeholder for accumulate_maxsumexp kernel test
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/kernel/accumulate_maxsumexp.hh"

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
using namespace nntile::kernel::accumulate_maxsumexp;

// Type to acquire reference values
using ref_t = double;

// Struct to hold test data and reference results
template<typename T>
struct TestData
{
    using Y = typename T::repr_t;
    Index nelems; // Number of (max,sumexp) pairs

    Y eps_check;

    std::vector<T> src_init;
    std::vector<T> dst_init;

    std::vector<T> dst_ref;
};

// Reference implementation of the accumulate maxsumexp operation
template<typename T>
void reference_accumulate_maxsumexp(TestData<T>& data)
{
    using Y = typename T::repr_t;
    if (data.nelems == 0)
    {
        return;
    }

    data.dst_ref = data.dst_init; // Copy initial destination

    for(Index i = 0; i < data.nelems; ++i)
    {
        const ref_t src_max = static_cast<Y>(data.src_init[2*i]);
        const ref_t src_sumexp = static_cast<Y>(data.src_init[2*i+1]);
        const ref_t dst_max = static_cast<Y>(data.dst_ref[2*i]);
        const ref_t dst_sumexp = static_cast<Y>(data.dst_ref[2*i+1]);

        // Do nothing if sum of exponents of source is zero
        if(src_sumexp != 0.0)
        {
            // Overwrite if old value of sum is zero
            if(dst_sumexp == 0.0)
            {
                data.dst_ref[2*i] = data.src_init[2*i];
                data.dst_ref[2*i+1] = data.src_init[2*i+1];
            }
            // Otherwise update based on maximum
            else if(dst_max < src_max)
            {
                const ref_t diff = dst_max - src_max;
                ref_t new_sumexp = src_sumexp + dst_sumexp * std::exp(diff);
                data.dst_ref[2*i+1] = static_cast<Y>(new_sumexp);
                data.dst_ref[2*i] = data.src_init[2*i];
            }
            else
            {
                const ref_t diff = src_max - dst_max;
                ref_t new_sumexp = dst_sumexp + src_sumexp * std::exp(diff);
                data.dst_ref[2*i+1] = static_cast<Y>(new_sumexp);
            }
        }
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
void generate_data(TestData<T>& data, Index nelems, DataGen strategy)
{
    using Y = typename T::repr_t;
    data.nelems = nelems;

    data.src_init.resize(2 * nelems);
    data.dst_init.resize(2 * nelems);
    data.dst_ref.resize(2 * nelems);

    switch(strategy)
    {
        // Non-random input generation
        case DataGen::PRESET:
            for(Index i = 0; i < nelems; ++i)
            {
                // Set src values - mix of positive and negative values
                const Y src_max = -2.0 + i * 0.5;
                data.src_init[2*i] = src_max;
                const Y src_sumexp = 0.1 + i * 0.1;
                data.src_init[2*i+1] = src_sumexp;

                // Set initial dst values
                const Y dst_max = -1.0 + i * 0.3;
                data.dst_init[2*i] = dst_max;
                const Y dst_sumexp = 0.05 + i * 0.05;
                data.dst_init[2*i+1] = dst_sumexp;
            }
            break;
        // Specific random initialization
        case DataGen::RANDOM:
            std::mt19937 gen(42);
            std::uniform_real_distribution<Y> dist_max(-3.0, 3.0);
            std::uniform_real_distribution<Y> dist_sumexp(0.01, 2.0);
            for(Index i = 0; i < nelems; ++i)
            {
                data.src_init[2*i] = dist_max(gen);
                data.src_init[2*i+1] = dist_sumexp(gen);
                data.dst_init[2*i] = dist_max(gen);
                data.dst_init[2*i+1] = dist_sumexp(gen);
            }
            break;
    }
}

// Get test input data (reference computation is done separately)
template<typename T>
TestData<T> get_test_input_data(Index nelems, DataGen strategy)
{
    using Y = typename T::repr_t;
    TestData<T> data;
    // Generate data by a provided strategy
    generate_data(data, nelems, strategy);

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
        data.eps_check = Y{1e-5};
    }
    else if (std::is_same_v<T, fp64_t>)
    {
        data.eps_check = Y{1e-12};
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

    // Check that source data was not modified
    for(Index i = 0; i < 2 * data.nelems; ++i)
    {
        REQUIRE(static_cast<Y>(src[i]) == static_cast<Y>(data.src_init[i]));
    }

    for(Index i = 0; i < data.nelems; ++i)
    {
        Y dst_max_ref = static_cast<Y>(data.dst_ref[2*i]);
        Y dst_sumexp_ref = static_cast<Y>(data.dst_ref[2*i+1]);

        REQUIRE(static_cast<Y>(dst[2*i]) == dst_max_ref);
        REQUIRE_THAT(
            static_cast<Y>(dst[2*i+1]),
            WithinRel(dst_sumexp_ref, data.eps_check)
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
            "[kernel][accumulate_maxsumexp][cpu][nelems=" +
            std::to_string(data.nelems) +
            "]"
        )
        {
            cpu<T>(data.nelems, &src_cpu[0], &dst_cpu[0]);
        };
    }
    else
    {
        cpu<T>(data.nelems, &src_cpu[0], &dst_cpu[0]);
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
        cudaMalloc(&dev_src, sizeof(T) * data.src_init.size()),
        "cudaMalloc dev_src"
    );
    CUDA_CHECK(
        cudaMalloc(&dev_dst, sizeof(T) * data.dst_init.size()),
        "cudaMalloc dev_dst"
    );

    std::vector<T> dst_cuda(data.dst_init);
    std::vector<T> src_cuda(data.src_init);

    CUDA_CHECK(
        cudaMemcpy(
            dev_src,
            &src_cuda[0],
            sizeof(T) * data.src_init.size(),
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy dev_src"
    );
    CUDA_CHECK(
        cudaMemcpy(
            dev_dst,
            &dst_cuda[0],
            sizeof(T) * data.dst_init.size(),
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy dev_dst"
    );

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream), "cudaStreamCreate");

    if constexpr (run_bench)
    {
        BENCHMARK(
            "[kernel][accumulate_maxsumexp][cuda][nelems=" +
            std::to_string(data.nelems) +
            "]"
        )
        {
            cuda<T>(stream, data.nelems, dev_src, dev_dst);
            cudaStreamSynchronize(stream);
        };
    }
    else
    {
        cuda<T>(stream, data.nelems, dev_src, dev_dst);
        CUDA_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

        CUDA_CHECK(
            cudaMemcpy(
                &dst_cuda[0],
                dev_dst,
                sizeof(T) * data.dst_init.size(),
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy dst_cuda"
        );
        CUDA_CHECK(
            cudaMemcpy(
                &src_cuda[0],
                dev_src,
                sizeof(T) * data.src_init.size(),
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
    "Accumulate MaxSumExp Kernel Verification",
    "[accumulate_maxsumexp]",
    fp64_t,
    fp32_t,
    fp16_t,
    bf16_t
)
{
    using T = TestType;
    const Index nelems = GENERATE(1, 5, 10, 20);
    const DataGen strategy = GENERATE(DataGen::PRESET, DataGen::RANDOM);

    auto data = get_test_input_data<T>(nelems, strategy);

    // Compute reference outputs for verification
    reference_accumulate_maxsumexp(data);

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
    "Accumulate MaxSumExp Kernel Benchmark",
    "[accumulate_maxsumexp][!benchmark]",
    fp64_t,
    fp32_t,
    fp16_t,
    bf16_t
)
{
    using T = TestType;
    const Index nelems = GENERATE(100, 1000, 10000);
    const DataGen strategy = GENERATE(DataGen::PRESET);

    auto data = get_test_input_data<T>(nelems, strategy);

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
