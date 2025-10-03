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

#ifdef NNTILE_USE_CUDA
template<typename T>
void run_cuda(Index m, Index n, Index k, Scalar alpha,
        const std::vector<T> &src, Scalar beta, std::vector<T> &dst)
{
    // Copy to device
    T *dev_src, *dev_dst;
    cudaError_t cuda_err = cudaMalloc(&dev_src, sizeof(T)*m*n);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMalloc(&dev_dst, sizeof(T)*m*n*k);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_src, &src[0], sizeof(T)*m*n,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_dst, &dst[0], sizeof(T)*m*n*k,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Init stream
    cudaStream_t stream;
    cuda_err = cudaStreamCreate(&stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Launch low-level CUDA kernel
    cuda<T>(stream, m, n, k, alpha, dev_src, beta, dev_dst);
    cuda_err = cudaStreamSynchronize(stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Copy result and deallocate device memory
    cuda_err = cudaMemcpy(&dst[0], dev_dst, sizeof(T)*m*n*k,
            cudaMemcpyDeviceToHost);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaFree(dev_src);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaFree(dev_dst);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaStreamDestroy(stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
}
#endif // NNTILE_USE_CUDA

// Templated validation
template<typename T>
void validate(Index m, Index n, Index k)
{
    using Y = typename T::repr_t;
    const Y eps = 2 * T::epsilon;
    // Init test input
    std::vector<T> src(m*n), dst(m*n*k);
    for(Index i0 = 0; i0 < m; ++i0)
    {
        for(Index i1 = 0; i1 < n; ++i1)
        {
            for(Index i2 = 0; i2 < k; ++i2)
            {
                dst[(i1*k+i2)*m+i0] = Y(i0+i1+i2) / Y{30};
            }
        }
    }
    for(Index i0 = 0; i0 < m; ++i0)
    {
        for(Index i1 = 0; i1 < n; ++i1)
        {
            src[i1*m+i0] = Y(i0+i1) / Y{20};
        }
    }
    // Save original dst
    std::vector<T> dst_save(dst);
    // Check low-level CPU kernel
    std::cout << "Run kernel::add_slice_inplace::cpu<" << T::short_name << ">\n";
    cpu<T>(m, n, k, -2.0, &src[0], 3.0, &dst[0]);
    for(Index i0 = 0; i0 < m; ++i0)
    {
        for(Index i1 = 0; i1 < n; ++i1)
        {
            // Check i2=0 at first, as it means val_ref = 0
            Y val{dst[i1*k*m+i0]};
            TEST_ASSERT(std::abs(val) <= eps);
            for(Index i2 = 1; i2 < k; ++i2)
            {
                Y val{dst[(i1*k+i2)*m+i0]};
                Y val_ref = Y(i2) / Y{10};
                TEST_ASSERT(std::abs(val/val_ref-Y{1}) <= 10*eps);
            }
        }
    }
    std::cout << "OK: kernel::add_slice_inplace::cpu<" << T::short_name << ">\n";
#ifdef NNTILE_USE_CUDA
    // Check low-level CUDA kernel
    dst = dst_save;
    std::cout << "Run kernel::add_slice_inplace::cuda<" << T::short_name << ">\n";
    run_cuda<T>(m, n, k, -2.0, src, 3.0, dst);
    for(Index i0 = 0; i0 < m; ++i0)
    {
        for(Index i1 = 0; i1 < n; ++i1)
        {
            // Check i2=0 at first, as it means val_ref = 0
            Y val{dst[i1*k*m+i0]};
            TEST_ASSERT(std::abs(val) <= 10*eps);
            for(Index i2 = 1; i2 < k; ++i2)
            {
                Y val{dst[(i1*k+i2)*m+i0]};
                Y val_ref = Y(i2) / Y{10};
                if(std::abs(val/val_ref-Y{1}) <= eps);
            }
        }
    }
    std::cout << "OK: kernel::add_slice_inplace::cuda<" << T::short_name << ">\n";
#endif // NNTILE_USE_CUDA
}

// Struct to hold test data and reference results
template<typename T>
struct TestData
{
    using Y = typename T::repr_t;
    Index m, n, k; // Tensor dimensions
    Scalar alpha, beta; // Scalar factors

    std::vector<T> src;
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
            Y src_val = static_cast<Y>(data.src[i1 * data.m + i0]);

            for(Index i2 = 0; i2 < data.k; ++i2)
            {
                Index dst_idx = (i1 * data.k + i2) * data.m + i0;
                Y& dst_val = reinterpret_cast<Y&>(data.dst_ref[dst_idx]);

                if(data.beta == 0.0)
                {
                    dst_val = data.alpha * src_val;
                }
                else
                {
                    dst_val = data.alpha * src_val + data.beta * dst_val;
                }
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

    data.src.resize(data.m * data.n);
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
                    data.src[i1 * data.m + i0] = Y(i0 + i1) / Y{20};
                }
            }
            for(Index i0 = 0; i0 < data.m; ++i0)
            {
                for(Index i1 = 0; i1 < data.n; ++i1)
                {
                    for(Index i2 = 0; i2 < data.k; ++i2)
                    {
                        Index dst_idx = (i1 * data.k + i2) * data.m + i0;
                        data.dst_init[dst_idx] = Y(i0 + i1 + i2) / Y{30};
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
                    data.src[i1 * data.m + i0] = dist(gen);
                }
            }
            for(Index i0 = 0; i0 < data.m; ++i0)
            {
                for(Index i1 = 0; i1 < data.n; ++i1)
                {
                    for(Index i2 = 0; i2 < data.k; ++i2)
                    {
                        Index dst_idx = (i1 * data.k + i2) * data.m + i0;
                        data.dst_init[dst_idx] = dist(gen);
                    }
                }
            }
            break;
    }
}

// Get test data and reference results
template<typename T>
TestData<T> get_test_data(Index m, Index n, Index k, Scalar alpha, Scalar beta, DataGen strategy)
{
    TestData<T> data;
    data.m = m;
    data.n = n;
    data.k = k;
    data.alpha = alpha;
    data.beta = beta;

    // Generate data by a provided strategy
    generate_data(data, strategy);

    // Compute reference outputs
    reference_add_slice_inplace(data);
    return data;
}

// Helper function to verify results
template<typename T>
void verify_results(const TestData<T>& data, const std::vector<T>& dst_out)
{
    using Y = typename T::repr_t;

    // Set accuracy threshold for each precision
    ref_t eps_check;
    if (std::is_same_v<T, bf16_t>)
    {
        eps_check = 1e-1;
    }
    else if (std::is_same_v<T, fp16_t>)
    {
        eps_check = 1e-2;
    }
    else if (std::is_same_v<T, fp32_t>)
    {
        eps_check = 1e-5;
    }
    else if (std::is_same_v<T, fp64_t>)
    {
        eps_check = 1e-12;
    }
    else
    {
        throw std::runtime_error("Unsupported data type");
    }

    for(Index i0 = 0; i0 < data.m; ++i0)
    {
        for(Index i1 = 0; i1 < data.n; ++i1)
        {
            Y src_val = static_cast<Y>(data.src[i1 * data.m + i0]);

            for(Index i2 = 0; i2 < data.k; ++i2)
            {
                Index dst_idx = (i1 * data.k + i2) * data.m + i0;
                Y dst_ref = reinterpret_cast<const Y&>(data.dst_ref[dst_idx]);

                REQUIRE_THAT(
                    static_cast<Y>(dst_out[dst_idx]),
                    WithinAbs(dst_ref, eps_check) || WithinRel(dst_ref, eps_check)
                );
            }
        }
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
            "[kernel][add_slice_inplace][cpu][m=" +
            std::to_string(data.m) + "][n=" + std::to_string(data.n) +
            "][k=" + std::to_string(data.k) +
            "][alpha=" + std::to_string(data.alpha) +
            "][beta=" + std::to_string(data.beta) + "]"
        )
        {
            cpu<T>(data.m, data.n, data.k, data.alpha, &data.src[0], data.beta, &dst_cpu[0]);
        };
    }
    else
    {
        cpu<T>(data.m, data.n, data.k, data.alpha, &data.src[0], data.beta, &dst_cpu[0]);
        verify_results(data, dst_cpu);
    }
}

#ifdef NNTILE_USE_CUDA

// Helper function to run CUDA test and verify results
template<typename T, bool run_bench>
void run_cuda_test(TestData<T>& data)
{
    T *dev_src, *dev_dst;
    CUDA_CHECK(cudaMalloc(&dev_src, sizeof(T) * data.m * data.n),
               "cudaMalloc dev_src");
    CUDA_CHECK(cudaMalloc(&dev_dst, sizeof(T) * data.m * data.n * data.k),
               "cudaMalloc dev_dst");

    std::vector<T> dst_cuda(data.dst_init);

    CUDA_CHECK(cudaMemcpy(dev_src, &data.src[0], sizeof(T) * data.m * data.n,
                          cudaMemcpyHostToDevice), "cudaMemcpy dev_src");
    CUDA_CHECK(cudaMemcpy(dev_dst, &dst_cuda[0], sizeof(T) * data.m * data.n * data.k,
                          cudaMemcpyHostToDevice), "cudaMemcpy dev_dst");

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
            cuda<T>(stream, data.m, data.n, data.k, data.alpha, dev_src, data.beta, dev_dst);
            cudaStreamSynchronize(stream);
        };
    }
    else
    {
        cuda<T>(stream, data.m, data.n, data.k, data.alpha, dev_src, data.beta, dev_dst);
        CUDA_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

        CUDA_CHECK(cudaMemcpy(&dst_cuda[0], dev_dst, sizeof(T) * data.m * data.n * data.k,
                              cudaMemcpyDeviceToHost), "cudaMemcpy dst_cuda");

        verify_results(data, dst_cuda);
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

    auto data = get_test_data<T>(m, n, k, alpha, beta, strategy);

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

    auto data = get_test_data<T>(m, n, k, alpha, beta, strategy);

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
