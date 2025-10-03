/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/add_fiber.cc
 * Per-element addition of a tensor and a broadcasted fiber
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/kernel/add_fiber.hh"

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
using namespace nntile::kernel::add_fiber;


#ifdef NNTILE_USE_CUDA
template<typename T>
void run_cuda(Index m, Index n, Index k, Index batch, Scalar alpha,
    const std::vector<T> &src1, Scalar beta, const std::vector<T> &src2,
    std::vector<T> &dst)
{
    // Copy to device
    T *dev_src1, *dev_src2, *dev_dst;
    cudaError_t cuda_err = cudaMalloc(&dev_src1, sizeof(T)*k*batch);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMalloc(&dev_src2, sizeof(T)*m*n*k*batch);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMalloc(&dev_dst, sizeof(T)*m*n*k*batch);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_src1, &src1[0], sizeof(T)*k*batch,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_src2, &src2[0], sizeof(T)*m*n*k*batch,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_dst, &dst[0], sizeof(T)*m*n*k*batch,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Init stream
    cudaStream_t stream;
    cuda_err = cudaStreamCreate(&stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Launch low-level CUDA kernel
    cuda<T>(stream, m, n, k, batch, alpha, dev_src1, beta, dev_src2, dev_dst);
    cuda_err = cudaStreamSynchronize(stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Copy result and deallocate device memory
    cuda_err = cudaMemcpy(&dst[0], dev_dst, sizeof(T)*m*n*k*batch,
            cudaMemcpyDeviceToHost);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaFree(dev_src1);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaFree(dev_src2);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaFree(dev_dst);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaStreamDestroy(stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
}
#endif // NNTILE_USE_CUDA

// Struct to hold test data and reference results
template<typename T>
struct TestData
{
    using Y = typename T::repr_t;
    Index m, n, k, batch; // Tensor dimensions
    Scalar alpha, beta;   // Scalar factors

    Y eps_check;

    std::vector<T> src1_init;
    std::vector<T> src2_init;
    std::vector<T> dst_init;

    std::vector<T> dst_ref;
};

// Reference implementation of the add fiber operation
template<typename T>
void reference_add_fiber(TestData<T>& data)
{
    using Y = typename T::repr_t;

    for(Index b = 0; b < data.batch; ++b)
    {
        for(Index i2 = 0; i2 < data.k; ++i2)
        {
            const Y src1_val = data.alpha * static_cast<Y>(data.src1_init[i2 + b * data.k]);

            for(Index i1 = 0; i1 < data.n; ++i1)
            {
                for(Index i0 = 0; i0 < data.m; ++i0)
                {
                    Index src2_idx = ((i1 + b * data.n) * data.k + i2) * data.m + i0;
                    Index dst_idx = ((i1 + b * data.n) * data.k + i2) * data.m + i0;

                    Y src2_val = static_cast<Y>(data.src2_init[src2_idx]);
                    Y& dst_val = reinterpret_cast<Y&>(data.dst_ref[dst_idx]);

                    if(std::abs(data.beta) <= Y(1e-6))
                    {
                        dst_val = src1_val;
                    }
                    else
                    {
                        dst_val = data.beta * src2_val + src1_val;
                    }
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

    data.src1_init_init.resize(data.k * data.batch);
    data.src2_init_init.resize(data.m * data.n * data.k * data.batch);
    data.dst_init.resize(data.m * data.n * data.k * data.batch);
    data.dst_ref.resize(data.m * data.n * data.k * data.batch);

    switch(strategy)
    {
        // Non-random input generation
        case DataGen::PRESET:
            for(Index i = 0; i < data.k * data.batch; ++i)
            {
                data.src1_init_init[i] = Y(2 * i + 1 - data.k * data.batch);
            }
            for(Index i = 0; i < data.m * data.n * data.k * data.batch; ++i)
            {
                data.src2_init_init[i] = Y(5 * data.m * data.n * data.k * data.batch - 2 * i);
                data.dst_init[i] = Y(3 * data.m * data.n * data.k * data.batch - i);
            }
            break;
        // Specific random initialization
        case DataGen::RANDOM:
            std::mt19937 gen(42);
            std::uniform_real_distribution<Y> dist(-2.0, 2.0);
            for(Index i = 0; i < data.k * data.batch; ++i)
            {
                data.src1_init_init[i] = dist(gen);
            }
            for(Index i = 0; i < data.m * data.n * data.k * data.batch; ++i)
            {
                data.src2_init_init[i] = dist(gen);
                data.dst_init[i] = dist(gen);
            }
            break;
    }
}

// Get test data and reference results
template<typename T>
TestData<T> get_test_data(Index m, Index n, Index k, Index batch,
                         Scalar alpha, Scalar beta, DataGen strategy)
{
    using Y = typename T::repr_t;
    TestData<T> data;
    data.m = m;
    data.n = n;
    data.k = k;
    data.batch = batch;
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
    reference_add_fiber(data);
    return data;
}

// Helper function to verify results
template<typename T>
void verify_results(const TestData<T>& data, const std::vector<T>& dst_out)
{
    using Y = typename T::repr_t;

    for(Index b = 0; b < data.batch; ++b)
    {
        for(Index i2 = 0; i2 < data.k; ++i2)
        {
            const Y src1_val = data.alpha * static_cast<Y>(data.src1_init[i2 + b * data.k]);

            for(Index i1 = 0; i1 < data.n; ++i1)
            {
                for(Index i0 = 0; i0 < data.m; ++i0)
                {
                    Index dst_idx = ((i1 + b * data.n) * data.k + i2) * data.m + i0;
                    Y dst_ref = reinterpret_cast<const Y&>(data.dst_ref[dst_idx]);

                    REQUIRE_THAT(
                        static_cast<Y>(dst_out[dst_idx]),
                        WithinAbs(dst_ref, data.eps_check) ||
                        WithinRel(dst_ref, data.eps_check)
                    );
                }
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
            "[kernel][add_fiber][cpu][m=" +
            std::to_string(data.m) + "][n=" + std::to_string(data.n) +
            "][k=" + std::to_string(data.k) + "][batch=" + std::to_string(data.batch) +
            "][alpha=" + std::to_string(data.alpha) + "][beta=" + std::to_string(data.beta) + "]"
        )
        {
            cpu<T>(data.m, data.n, data.k, data.batch, data.alpha, &data.src1_init[0], data.beta, &data.src2_init[0], &dst_cpu[0]);
        };
    }
    else
    {
        cpu<T>(data.m, data.n, data.k, data.batch, data.alpha, &data.src1_init[0], data.beta, &data.src2_init[0], &dst_cpu[0]);
        verify_results(data, dst_cpu);
    }
}

#ifdef NNTILE_USE_CUDA

// Helper function to run CUDA test and verify results
template<typename T, bool run_bench>
void run_cuda_test(TestData<T>& data)
{
    T *dev_src1, *dev_src2, *dev_dst;
    CUDA_CHECK(cudaMalloc(&dev_src1, sizeof(T) * data.k * data.batch),
               "cudaMalloc dev_src1");
    CUDA_CHECK(cudaMalloc(&dev_src2, sizeof(T) * data.m * data.n * data.k * data.batch),
               "cudaMalloc dev_src2");
    CUDA_CHECK(cudaMalloc(&dev_dst, sizeof(T) * data.m * data.n * data.k * data.batch),
               "cudaMalloc dev_dst");

    std::vector<T> dst_cuda(data.m * data.n * data.k * data.batch);

    CUDA_CHECK(cudaMemcpy(dev_src1, &data.src1_init[0], sizeof(T) * data.k * data.batch,
                          cudaMemcpyHostToDevice), "cudaMemcpy dev_src1");
    CUDA_CHECK(cudaMemcpy(dev_src2, &data.src2_init[0], sizeof(T) * data.m * data.n * data.k * data.batch,
                          cudaMemcpyHostToDevice), "cudaMemcpy dev_src2");
    CUDA_CHECK(cudaMemcpy(dev_dst, &dst_cuda[0], sizeof(T) * data.m * data.n * data.k * data.batch,
                          cudaMemcpyHostToDevice), "cudaMemcpy dev_dst");

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream), "cudaStreamCreate");

    if constexpr (run_bench)
    {
        BENCHMARK(
            "[kernel][add_fiber][cuda][m=" +
            std::to_string(data.m) + "][n=" + std::to_string(data.n) +
            "][k=" + std::to_string(data.k) + "][batch=" + std::to_string(data.batch) +
            "][alpha=" + std::to_string(data.alpha) + "][beta=" + std::to_string(data.beta) + "]"
        )
        {
            cuda<T>(stream, data.m, data.n, data.k, data.batch, data.alpha, dev_src1, data.beta, dev_src2, dev_dst);
            cudaStreamSynchronize(stream);
        };
    }
    else
    {
        cuda<T>(stream, data.m, data.n, data.k, data.batch, data.alpha, dev_src1, data.beta, dev_src2, dev_dst);
        CUDA_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

        CUDA_CHECK(cudaMemcpy(&dst_cuda[0], dev_dst, sizeof(T) * data.m * data.n * data.k * data.batch,
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
    "Add Fiber Kernel Verification",
    "[add_fiber]",
    fp64_t,
    fp32_t,
    fp16_t,
    bf16_t
)
{
    using T = TestType;
    const Index m = GENERATE(1, 5);
    const Index n = GENERATE(1, 3);
    const Index k = GENERATE(1, 10);
    const Index batch = GENERATE(1, 4);
    const Scalar alpha = GENERATE(0.5, 1.0, 2.0);
    const Scalar beta = GENERATE(0.0, 0.5, -1.0);
    const DataGen strategy = GENERATE(DataGen::PRESET, DataGen::RANDOM);

    auto data = get_test_data<T>(m, n, k, batch, alpha, beta, strategy);

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
    "Add Fiber Kernel Benchmark",
    "[add_fiber][!benchmark]",
    fp64_t,
    fp32_t,
    fp16_t,
    bf16_t
)
{
    using T = TestType;
    const Index m = GENERATE(64, 256);
    const Index n = GENERATE(64, 256);
    const Index k = GENERATE(32, 128);
    const Index batch = GENERATE(4, 16);
    const Scalar alpha = GENERATE(1.0);
    const Scalar beta = GENERATE(-1.0);
    const DataGen strategy = GENERATE(DataGen::PRESET);

    auto data = get_test_data<T>(m, n, k, batch, alpha, beta, strategy);

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
