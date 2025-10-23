/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/add_slice_inplace.cc
 * Per-element addition of a tensor and a broadcasted slice
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/kernel/add_slice_inplace.hh"

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
using namespace nntile::kernel::add_slice_inplace;

// Type to acquire reference values
using ref_t = double;

// Struct to hold test data and reference results
template<typename T>
struct TestData
{
    using Y = typename T::repr_t;
    Index m, n, k; // Tensor dimensions
    Scalar alpha, beta; // Scalar factors

    Y eps_check;

    std::vector<T> src_init;
    std::vector<T> dst_init;

    std::vector<T> dst_ref;
};

// Reference implementation of the add slice inplace operation
template<typename T>
void reference_add_slice_inplace(TestData<T>& data)
{
    using Y = typename T::repr_t;

    data.dst_ref = data.dst_init; // Copy initial destination

    for(Index i0 = 0; i0 < data.m; ++i0)
    {
        for(Index i1 = 0; i1 < data.n; ++i1)
        {
            ref_t src_val = static_cast<Y>(data.src_init[i1 * data.m + i0]);
            Index dst_idx_base = i1 * data.k * data.m + i0;
            for(Index i2 = 0; i2 < data.k; ++i2)
            {
                Index dst_idx = i2 * data.m + dst_idx_base;
                ref_t dst_val = static_cast<Y>(data.dst_ref[dst_idx]);

                if(data.beta == 0.0)
                {
                    dst_val = data.alpha * src_val;
                }
                else
                {
                    dst_val = data.alpha * src_val + data.beta * dst_val;
                }

                data.dst_ref[dst_idx] = static_cast<Y>(dst_val);
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
void generate_data(TestData<T>& data, DataGen strategy)
{
    using Y = typename T::repr_t;

    data.src_init.resize(data.m * data.n);
    data.dst_init.resize(data.m * data.n * data.k);
    data.dst_ref.resize(data.m * data.n * data.k);

    switch(strategy)
    {
        // Non-random input generation
        case DataGen::PRESET:
            for(Index i0 = 0; i0 < data.m; ++i0)
            {
                for(Index i1 = 0; i1 < data.n; ++i1)
                {
                    const Y src_val = Y(i0 + i1) / Y(20);
                    data.src_init[i1 * data.m + i0] = src_val;
                }
            }
            for(Index i0 = 0; i0 < data.m; ++i0)
            {
                for(Index i1 = 0; i1 < data.n; ++i1)
                {
                    Index dst_idx_base = i1 * data.k * data.m + i0;
                    for(Index i2 = 0; i2 < data.k; ++i2)
                    {
                        Index dst_idx = i2 * data.m + dst_idx_base;
                        const Y dst_val = Y(i0 + i1 + i2) / Y(30);
                        data.dst_init[dst_idx] = dst_val;
                    }
                }
            }
            break;
        // Specific random initialization
        case DataGen::RANDOM:
            std::mt19937 gen(42);
            std::uniform_real_distribution<Y> dist(-2.0, 2.0);
            for(Index i0 = 0; i0 < data.m; ++i0)
            {
                for(Index i1 = 0; i1 < data.n; ++i1)
                {
                    data.src_init[i1 * data.m + i0] = dist(gen);
                }
            }
            for(Index i0 = 0; i0 < data.m; ++i0)
            {
                for(Index i1 = 0; i1 < data.n; ++i1)
                {
                    Index dst_idx_base = i1 * data.k * data.m + i0;
                    for(Index i2 = 0; i2 < data.k; ++i2)
                    {
                        Index dst_idx = i2 * data.m + dst_idx_base;
                        data.dst_init[dst_idx] = dist(gen);
                    }
                }
            }
            break;
    }
}

// Get test input data (reference computation is done separately)
template<typename T>
TestData<T> get_test_input_data(
    Index m,
    Index n,
    Index k,
    Scalar alpha,
    Scalar beta,
    DataGen strategy
)
{
    using Y = typename T::repr_t;
    TestData<T> data;
    data.m = m;
    data.n = n;
    data.k = k;
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
    for(Index i = 0; i < data.m * data.n; ++i)
    {
        REQUIRE(static_cast<Y>(src[i]) == static_cast<Y>(data.src_init[i]));
    }

    for(Index i = 0; i < data.m * data.n * data.k; ++i)
    {
        REQUIRE_THAT(
            static_cast<Y>(dst[i]),
            WithinRel(static_cast<Y>(data.dst_ref[i]), data.eps_check) ||
            WithinAbs(static_cast<Y>(data.dst_ref[i]), data.eps_check)
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
            "[kernel][add_slice_inplace][cpu][m=" +
            std::to_string(data.m) + "][n=" + std::to_string(data.n) +
            "][k=" + std::to_string(data.k) +
            "][alpha=" + std::to_string(data.alpha) +
            "][beta=" + std::to_string(data.beta) + "]"
        )
        {
            cpu<T>(
                data.m,
                data.n,
                data.k,
                data.alpha,
                &src_cpu[0],
                data.beta,
                &dst_cpu[0]
            );
        };
    }
    else
    {
        cpu<T>(
            data.m,
            data.n,
            data.k,
            data.alpha,
            &src_cpu[0],
            data.beta,
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
            "[kernel][add_slice_inplace][cuda][m=" +
            std::to_string(data.m) + "][n=" + std::to_string(data.n) +
            "][k=" + std::to_string(data.k) +
            "][alpha=" + std::to_string(data.alpha) +
            "][beta=" + std::to_string(data.beta) + "]"
        )
        {
            cuda<T>(
                stream,
                data.m,
                data.n,
                data.k,
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
        cuda<T>(
            stream,
            data.m,
            data.n,
            data.k,
            data.alpha,
            dev_src,
            data.beta,
            dev_dst
        );
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
    "Add Slice Inplace Kernel Verification",
    "[add_slice_inplace]",
    fp64_t,
    fp32_t,
    fp16_t,
    bf16_t
)
{
    using T = TestType;
    const Index m = GENERATE(1, 8, 4);
    const Index n = GENERATE(9, 9, 7);
    const Index k = GENERATE(10, 1, 8);
    const Scalar alpha = GENERATE(-2.0, 1.0);
    const Scalar beta = GENERATE(3.0, 1.0);
    const DataGen strategy = GENERATE(DataGen::PRESET, DataGen::RANDOM);

    auto data = get_test_input_data<T>(m, n, k, alpha, beta, strategy);

    // Compute reference outputs for verification
    reference_add_slice_inplace(data);

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
    "Add Slice Inplace Kernel Benchmark",
    "[add_slice_inplace][!benchmark]",
    fp64_t,
    fp32_t,
    fp16_t,
    bf16_t
)
{
    using T = TestType;
    const Index m = GENERATE(64, 128);
    const Index n = GENERATE(64, 128);
    const Index k = GENERATE(32, 64);
    const Scalar alpha = GENERATE(1.0);
    const Scalar beta = GENERATE(1.0);
    const DataGen strategy = GENERATE(DataGen::PRESET);

    auto data = get_test_input_data<T>(m, n, k, alpha, beta, strategy);

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
