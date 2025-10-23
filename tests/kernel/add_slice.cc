/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/add_slice.cc
 * Per-element addition of a tensor and a broadcasted slice
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/kernel/add_slice.hh"

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
using namespace nntile::kernel::add_slice;

// Type to acquire reference values
using ref_t = double;

// Struct to hold test data and reference results
template<typename T>
struct TestData
{
    using Y = typename T::repr_t;
    Index m, n, k; // Dimensions
    Scalar alpha, beta;

    Y eps_check;

    std::vector<T> src1;
    std::vector<T> src2;
    std::vector<T> dst_init;
    std::vector<T> dst_ref;
};

// Reference implementation of the add_slice operation
template<typename T>
void reference_add_slice(TestData<T>& data)
{
    using Y = typename T::repr_t;
    if (data.m == 0 || data.n == 0 || data.k == 0)
    {
        return;
    }
    const ref_t alpha_r = data.alpha;
    const ref_t beta_r = data.beta;

    for(Index i2 = 0; i2 < data.n; ++i2)
    {
        for(Index i1 = 0; i1 < data.m; ++i1)
        {
            ref_t src1_v = static_cast<Y>(data.src1[i2*data.m+i1]);
            src1_v *= alpha_r;
            for(Index i0 = 0; i0 < data.k; ++i0)
            {
                T src2_v_T = data.src2[i2*data.m*data.k + i0*data.m + i1];
                ref_t src2_v = static_cast<Y>(src2_v_T);
                src2_v *= beta_r;
                T dst_v_T = static_cast<T>(static_cast<Y>(src1_v + src2_v));
                data.dst_ref[i2*data.m*data.k + i0*data.m + i1] = dst_v_T;
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

    switch(strategy)
    {
        // Non-random input generation
        case DataGen::PRESET:
            for(Index i = 0; i < data.m * data.n; ++i)
            {
                data.src1[i] = Y(2 * i + 1);
            }
            for(Index i = 0; i < data.m * data.k * data.n; ++i)
            {
                data.src2[i] = Y(i + 1);
                data.dst_init[i] = Y(i - 1);
            }
            break;
        // Specific random initialization
        case DataGen::RANDOM:
            std::mt19937 gen(42);
            std::uniform_real_distribution<Y> dist(1.0, 2.0);
            for(Index i = 0; i < data.m * data.n; ++i)
            {
                data.src1[i] = dist(gen);
            }
            for(Index i = 0; i < data.m * data.k * data.n; ++i)
            {
                data.src2[i] = dist(gen);
                data.dst_init[i] = dist(gen);
            }
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
    TestData<T> data;
    // Generate data by a provided strategy
    data.m = m;
    data.n = n;
    data.k = k;
    data.alpha = alpha;
    data.beta = beta;
    data.src1.resize(m * n);
    data.src2.resize(m * k * n);
    data.dst_init.resize(m * k * n);
    data.dst_ref.resize(m * k * n);
    generate_data(data, strategy);

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
    for(Index i = 0; i < data.m * data.k * data.n; ++i)
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
    std::vector<T> dst_cpu(data.dst_init);

    if constexpr (run_bench)
    {
        BENCHMARK(
            "[kernel][add_slice][cpu][m=" +
            std::to_string(data.m) +
            "][n=" +
            std::to_string(data.n) +
            "][k=" +
            std::to_string(data.k) +
            "]"
        )
        {
            cpu<T>(
                data.m,
                data.n,
                data.k,
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
            data.m,
            data.n,
            data.k,
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
    CUDA_CHECK(cudaMalloc(&dev_src1, sizeof(T) * data.m * data.n),
               "cudaMalloc dev_src1");
    CUDA_CHECK(cudaMalloc(&dev_src2, sizeof(T) * data.m * data.k * data.n),
               "cudaMalloc dev_src2");
    CUDA_CHECK(cudaMalloc(&dev_dst, sizeof(T) * data.m * data.k * data.n),
               "cudaMalloc dev_dst");

    std::vector<T> dst_cuda(data.dst_init);

    CUDA_CHECK(
        cudaMemcpy(
            dev_src1,
            &data.src1[0],
            sizeof(T) * data.m * data.n,
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy dev_src1"
    );
    CUDA_CHECK(
        cudaMemcpy(
            dev_src2,
            &data.src2[0],
            sizeof(T) * data.m * data.k * data.n,
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy dev_src2"
    );
    CUDA_CHECK(
        cudaMemcpy(
            dev_dst,
            &dst_cuda[0],
            sizeof(T) * data.m * data.k * data.n,
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy dev_dst"
    );

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream), "cudaStreamCreate");

    if constexpr (run_bench)
    {
        BENCHMARK(
            "[kernel][add_slice][cuda][m=" +
            std::to_string(data.m) +
            "][n=" +
            std::to_string(data.n) +
            "][k=" +
            std::to_string(data.k) +
            "]"
        )
        {
            cuda<T>(
                stream,
                data.m,
                data.n,
                data.k,
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
            data.m,
            data.n,
            data.k,
            data.alpha,
            dev_src1,
            data.beta,
            dev_src2,
            dev_dst
        );
        CUDA_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

        CUDA_CHECK(
            cudaMemcpy(
                &dst_cuda[0],
                dev_dst,
                sizeof(T) * data.m * data.k * data.n,
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy dst_cuda"
        );

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
    "Add Slice Kernel Verification",
    "[add_slice]",
    fp64_t,
    fp32_t,
    fp16_t,
    bf16_t
)
{
    using T = TestType;
    const Index m = GENERATE(5, 19);
    const Index n = GENERATE(5, 16);
    const Index k = GENERATE(5, 21);
    const Scalar alpha = GENERATE(1.0, 0.5, 2.0);
    const Scalar beta = GENERATE(1.0, 0.5, 2.0);
    const DataGen strategy = GENERATE(DataGen::PRESET, DataGen::RANDOM);

    auto data = get_test_input_data<T>(
        m,
        n,
        k,
        alpha,
        beta,
        strategy
    );

    // Compute reference outputs for verification
    reference_add_slice(data);

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
    "Add Slice Kernel Benchmark",
    "[add_slice][!benchmark]",
    fp64_t,
    fp32_t,
    fp16_t,
    bf16_t
)
{
    using T = TestType;
    const Index m = GENERATE(512, 1024);
    const Index n = GENERATE(512, 1024);
    const Index k = GENERATE(512, 1024);
    const Scalar alpha = GENERATE(1.0);
    const Scalar beta = GENERATE(1.0);
    const DataGen strategy = GENERATE(DataGen::PRESET);

    auto data = get_test_input_data<T>(
        m,
        n,
        k,
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
