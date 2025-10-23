/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/maxsumexp.cc
 * Max and sum of exponents of a buffer
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/kernel/maxsumexp.hh"

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
using namespace nntile::kernel::maxsumexp;

// Type to acquire reference values
using ref_t = double;

// Struct to hold test data and reference results
template<typename T>
struct TestData
{
    using Y = typename T::repr_t;
    Index m; // Size of the first mode of src and the second mode of maxsumexp arrays
    Index n; // Size of the last mode of src and maxsumexp arrays
    Index k; // Size of the middle mode of src array

    Y eps_check;

    std::vector<T> src_init;
    std::vector<T> maxsumexp_init;

    std::vector<T> maxsumexp_ref;
};

// Reference implementation of the maxsumexp operation
template<typename T>
void reference_maxsumexp(TestData<T>& data)
{
    using Y = typename T::repr_t;
    if (data.m == 0 || data.n == 0 || data.k == 0)
    {
        return;
    }

    const Index mk = data.m * data.k;
    const Index dst_size = 2 * data.m * data.n;

    // Initialize reference output with initial maxsumexp values
    for(Index i = 0; i < dst_size; ++i)
    {
        data.maxsumexp_ref[i] = data.maxsumexp_init[i];
    }

    Index dst_offset = 0;
    // Cycle over n (last mode)
    for(Index i2 = 0; i2 < data.n; ++i2)
    {
        // Cycle over m (first mode)
        for(Index i1 = 0; i1 < data.m; ++i1)
        {
            // Get max and sum of exponents of a corresponding slice
            const Index src_slice_start = i2 * mk + i1;
            // Init max and sum with the first value
            ref_t max_val = static_cast<Y>(data.src_init[src_slice_start]);
            ref_t sum_exp = 1.0;
            bool has_finite_values = !std::isinf(max_val);

            // Cycle over slice of input buffer (k dimension)
            for(Index i0 = 1; i0 < data.k; ++i0)
            {
                const Index src_idx = src_slice_start + i0 * data.m;
                ref_t val = static_cast<Y>(data.src_init[src_idx]);
                // Ignore -inf value, which comes from mask
                if(std::isinf(val))
                {
                    continue;
                }
                has_finite_values = true;

                // Update max and sum of exponents
                if(max_val < val)
                {
                    sum_exp = sum_exp * std::exp(max_val - val) + 1.0;
                    max_val = val;
                }
                else
                {
                    sum_exp += std::exp(val - max_val);
                }
            }

            // Update result if we have finite values
            if(has_finite_values)
            {
                ref_t sum_old = static_cast<Y>(data.maxsumexp_ref[dst_offset+1]);
                // If old sum is zero then just overwrite it with current sum
                if(sum_old == 0.0)
                {
                    data.maxsumexp_ref[dst_offset] = static_cast<Y>(max_val);
                    data.maxsumexp_ref[dst_offset+1] = static_cast<Y>(sum_exp);
                }
                // Update non-zero initial sum
                else
                {
                    ref_t max_old = static_cast<Y>(data.maxsumexp_ref[dst_offset]);
                    if(max_old < max_val)
                    {
                        data.maxsumexp_ref[dst_offset] = static_cast<Y>(max_val);
                        data.maxsumexp_ref[dst_offset+1] = static_cast<Y>(sum_old * std::exp(max_old - max_val) + sum_exp);
                    }
                    else
                    {
                        data.maxsumexp_ref[dst_offset+1] = static_cast<Y>(sum_exp * std::exp(max_val - max_old) + sum_old);
                    }
                }
            }
            dst_offset += 2;
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

    const Index src_size = m * k * n;
    const Index maxsumexp_size = 2 * m * n;

    data.src_init.resize(src_size);
    data.maxsumexp_init.resize(maxsumexp_size);
    data.maxsumexp_ref.resize(maxsumexp_size);

    switch(strategy)
    {
        // Non-random input generation
        case DataGen::PRESET:
            for(Index i = 0; i < src_size; ++i)
            {
                const Y val = 2 * i + 1 - src_size;
                data.src_init[i] = val;
            }
            for(Index i = 0; i < maxsumexp_size; ++i)
            {
                const Y val = 5 * maxsumexp_size - 2 * i;
                data.maxsumexp_init[i] = val;
            }
            break;
        // Specific random initialization
        case DataGen::RANDOM:
            std::mt19937 gen(42);
            std::uniform_real_distribution<Y> dist(0.0, 1.0);
            for(Index i = 0; i < src_size; ++i)
            {
                data.src_init[i] = dist(gen);
            }
            for(Index i = 0; i < maxsumexp_size; ++i)
            {
                data.maxsumexp_init[i] = dist(gen);
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
    DataGen strategy
)
{
    TestData<T> data;
    // Generate data by a provided strategy
    generate_data(data, m, n, k, strategy);

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
    const std::vector<T>& src,
    const std::vector<T>& maxsumexp
)
{
    using Y = typename T::repr_t;

    // Check that src was not changed during kernel execution
    for(Index i = 0; i < data.src_init.size(); ++i)
    {
        REQUIRE(src[i].value == data.src_init[i].value);
    }

    // Check that maxsumexp (output) matches reference
    for(Index i = 0; i < data.maxsumexp_ref.size(); ++i)
    {
        const Y maxsumexp_ref = static_cast<Y>(data.maxsumexp_ref[i]);
        REQUIRE_THAT(
            static_cast<Y>(maxsumexp[i]),
            WithinRel(maxsumexp_ref, data.eps_check)
        );
    }
}

// Helper function to run CPU test and verify results
template<typename T, bool run_bench>
void run_cpu_test(TestData<T>& data)
{
    std::vector<T> maxsumexp_cpu(data.maxsumexp_init);
    std::vector<T> src_cpu(data.src_init);

    if constexpr (run_bench)
    {
        BENCHMARK(
            "[kernel][maxsumexp][cpu][m=" +
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
                &src_cpu[0],
                &maxsumexp_cpu[0]
            );
        };
    }
    else
    {
        cpu<T>(
            data.m,
            data.n,
            data.k,
            &src_cpu[0],
            &maxsumexp_cpu[0]
        );
        verify_results(data, src_cpu, maxsumexp_cpu);
    }
}

#ifdef NNTILE_USE_CUDA

// Helper function to run CUDA test and verify results
template<typename T, bool run_bench>
void run_cuda_test(TestData<T>& data)
{
    T *dev_src, *dev_maxsumexp;
    CUDA_CHECK(
        cudaMalloc(
            &dev_src,
            sizeof(T) * data.src_init.size()
        ),
        "cudaMalloc dev_src"
    );
    CUDA_CHECK(
        cudaMalloc(
            &dev_maxsumexp,
            sizeof(T) * data.maxsumexp_init.size()
        ),
        "cudaMalloc dev_maxsumexp"
    );

    std::vector<T> maxsumexp_cuda(data.maxsumexp_init);
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
            dev_maxsumexp,
            &maxsumexp_cuda[0],
            sizeof(T) * data.maxsumexp_init.size(),
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy dev_maxsumexp"
    );

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream), "cudaStreamCreate");

    if constexpr (run_bench)
    {
        BENCHMARK(
            "[kernel][maxsumexp][cuda][m=" +
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
                dev_src,
                dev_maxsumexp
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
            dev_src,
            dev_maxsumexp
        );
        CUDA_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

        CUDA_CHECK(
            cudaMemcpy(
                &maxsumexp_cuda[0],
                dev_maxsumexp,
                sizeof(T) * data.maxsumexp_init.size(),
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy maxsumexp_cuda"
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

        verify_results(data, src_cuda, maxsumexp_cuda);
    }

    CUDA_CHECK(cudaFree(dev_src), "cudaFree dev_src");
    CUDA_CHECK(cudaFree(dev_maxsumexp), "cudaFree dev_maxsumexp");
    CUDA_CHECK(cudaStreamDestroy(stream), "cudaStreamDestroy");
}
#endif

// Catch2-based tests
TEMPLATE_TEST_CASE(
    "MaxSumExp Kernel Verification",
    "[maxsumexp]",
    fp64_t,
    fp32_t,
    fp16_t,
    bf16_t
)
{
    using T = TestType;
    const Index m = GENERATE(3, 8);
    const Index n = GENERATE(4, 9);
    const Index k = GENERATE(5, 10);
    const DataGen strategy = GENERATE(DataGen::PRESET, DataGen::RANDOM);

    auto data = get_test_input_data<T>(
        m,
        n,
        k,
        strategy
    );

    // Compute reference outputs for verification
    reference_maxsumexp(data);

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
    "MaxSumExp Kernel Benchmark",
    "[maxsumexp][!benchmark]",
    fp64_t,
    fp32_t,
    fp16_t,
    bf16_t
)
{
    using T = TestType;
    const Index m = GENERATE(64, 256);
    const Index n = GENERATE(64, 256);
    const Index k = GENERATE(128, 512);
    const DataGen strategy = GENERATE(DataGen::PRESET);

    auto data = get_test_input_data<T>(
        m,
        n,
        k,
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
