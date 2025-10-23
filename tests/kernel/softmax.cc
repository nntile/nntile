/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/softmax.cc
 * Softmax operation on a buffer
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/kernel/softmax.hh"

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
using namespace nntile::kernel::softmax;

// Type to acquire reference values
using ref_t = double;

// Struct to hold test data and reference results
template<typename T>
struct TestData
{
    using Y = typename T::repr_t;
    Index m; // First mode size
    Index n; // Last mode size
    Index k; // Middle mode size
    Scalar alpha;

    Y eps_check;

    std::vector<T> maxsumexp; // Size: 2*m*n (interleaved max and sumexp)
    std::vector<T> src;       // Size: m*n*k
    std::vector<T> dst_init;  // Size: m*n*k
    std::vector<T> dst_ref;   // Size: m*n*k
};

// Reference implementation of the softmax operation
template<typename T>
void reference_softmax(TestData<T>& data)
{
    using Y = typename T::repr_t;
    const ref_t alpha = data.alpha;
    Index src_dst_offset = 0;

    // Outer loop by the last mode
    for(Index i2 = 0; i2 < data.n; ++i2)
    {
        // Middle loop by the middle mode
        for(Index i1 = 0; i1 < data.k; ++i1)
        {
            Index maxsumexp_offset = 2 * data.m * i2;
            // Inner loop by the first mode
            for(Index i0 = 0; i0 < data.m; ++i0)
            {
                // Value-to-update
                ref_t val = static_cast<Y>(data.src[src_dst_offset]);
                // Max and sum of exponents
                const ref_t max = static_cast<Y>(data.maxsumexp[maxsumexp_offset]);
                const ref_t sum = static_cast<Y>(data.maxsumexp[maxsumexp_offset+1]);
                // Update value
                ref_t result = 0.0;
                if(not std::isinf(val))
                {
                    result = alpha * std::exp(val - max) / sum;
                }
                data.dst_ref[src_dst_offset] = static_cast<T>(static_cast<Y>(result));
                // Update pointers
                ++src_dst_offset;
                maxsumexp_offset += 2;
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
void generate_data(TestData<T>& data, Index m, Index n, Index k, DataGen strategy)
{
    using Y = typename T::repr_t;
    data.m = m;
    data.n = n;
    data.k = k;

    data.maxsumexp.resize(2 * m * n);
    data.src.resize(m * n * k);
    data.dst_ref.resize(m * n * k);
    data.dst_init.resize(m * n * k);

    switch(strategy)
    {
        // Non-random input generation
        case DataGen::PRESET:
            for(Index i = 0; i < 2 * m * n; i += 2)
            {
                data.maxsumexp[i] = Y(1.0);     // max value
                data.maxsumexp[i+1] = Y(10.0);  // sum of exponents
            }
            for(Index i = 0; i < m * n * k; ++i)
            {
                Y src_val = Y(i % 10 - 5);
                if(src_val < -3)
                {
                    src_val = -std::numeric_limits<Y>::infinity();
                }
                data.src[i] = src_val;
                data.dst_init[i] = Y(i % 3 - 1);
            }
            break;
        // Specific random initialization
        case DataGen::RANDOM:
            std::mt19937 gen(42);
            std::uniform_real_distribution<Y> dist_max(-1.0, 1.0);
            std::uniform_real_distribution<Y> dist_sum(5.0, 15.0);
            std::uniform_real_distribution<Y> dist_src(-2.0, 2.0);
            for(Index i = 0; i < 2 * m * n; i += 2)
            {
                data.maxsumexp[i] = dist_max(gen);     // max value
                data.maxsumexp[i+1] = dist_sum(gen);   // sum of exponents
            }
            for(Index i = 0; i < m * n * k; ++i)
            {
                Y src_val = dist_src(gen);
                if(src_val < -1)
                {
                    src_val = -std::numeric_limits<Y>::infinity();
                }
                data.src[i] = src_val;
                data.dst_init[i] = dist_src(gen);
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
    DataGen strategy
)
{
    TestData<T> data;
    // Generate data by a provided strategy
    generate_data(data, m, n, k, strategy);
    // Fill in remaining fields of TestData
    data.alpha = alpha;
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
    for(Index i = 0; i < data.m * data.n * data.k; ++i)
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
            "[kernel][softmax][cpu][m=" +
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
                &data.maxsumexp[0],
                &data.src[0],
                data.alpha,
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
            &data.maxsumexp[0],
            &data.src[0],
            data.alpha,
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
    T *dev_maxsumexp, *dev_src, *dev_dst;
    CUDA_CHECK(cudaMalloc(&dev_maxsumexp, sizeof(T) * 2 * data.m * data.n),
               "cudaMalloc dev_maxsumexp");
    CUDA_CHECK(cudaMalloc(&dev_src, sizeof(T) * data.m * data.n * data.k),
               "cudaMalloc dev_src");
    CUDA_CHECK(cudaMalloc(&dev_dst, sizeof(T) * data.m * data.n * data.k),
               "cudaMalloc dev_dst");

    std::vector<T> dst_cuda(data.dst_init);

    CUDA_CHECK(cudaMemcpy(dev_maxsumexp, &data.maxsumexp[0],
                          sizeof(T) * 2 * data.m * data.n,
                          cudaMemcpyHostToDevice), "cudaMemcpy dev_maxsumexp");
    CUDA_CHECK(cudaMemcpy(dev_src, &data.src[0],
                          sizeof(T) * data.m * data.n * data.k,
                          cudaMemcpyHostToDevice), "cudaMemcpy dev_src");

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream), "cudaStreamCreate");

    if constexpr (run_bench)
    {
        BENCHMARK(
            "[kernel][softmax][cuda][m=" +
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
                dev_maxsumexp,
                dev_src,
                data.alpha,
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
            dev_maxsumexp,
            dev_src,
            data.alpha,
            dev_dst
        );
        CUDA_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

        CUDA_CHECK(cudaMemcpy(&dst_cuda[0], dev_dst,
                              sizeof(T) * data.m * data.n * data.k,
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy dst_cuda");

        verify_results(data, dst_cuda);
    }

    CUDA_CHECK(cudaFree(dev_maxsumexp), "cudaFree dev_maxsumexp");
    CUDA_CHECK(cudaFree(dev_src), "cudaFree dev_src");
    CUDA_CHECK(cudaFree(dev_dst), "cudaFree dev_dst");
    CUDA_CHECK(cudaStreamDestroy(stream), "cudaStreamDestroy");
}
#endif // NNTILE_USE_CUDA

// Catch2-based tests
TEMPLATE_TEST_CASE(
    "Softmax Kernel Verification",
    "[softmax]",
    fp64_t,
    fp32_t,
    fp16_t,
    bf16_t
)
{
    using T = TestType;
    const Index m = GENERATE(3, 45);
    const Index n = GENERATE(2, 49);
    const Index k = GENERATE(4, 37);
    const Scalar alpha = GENERATE(1.0, 2.0);
    const DataGen strategy = GENERATE(DataGen::PRESET, DataGen::RANDOM);

    auto data = get_test_input_data<T>(
        m, n, k,
        alpha,
        strategy
    );

    // Compute reference outputs for verification
    reference_softmax(data);

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
    "Softmax Kernel Benchmark",
    "[softmax][!benchmark]",
    fp64_t,
    fp32_t,
    fp16_t,
    bf16_t
)
{
    using T = TestType;
    const Index m = GENERATE(128);
    const Index n = GENERATE(128);
    const Index k = GENERATE(128);
    const Scalar alpha = GENERATE(1.0);
    const DataGen strategy = GENERATE(DataGen::PRESET);

    auto data = get_test_input_data<T>(
        m, n, k,
        alpha,
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
