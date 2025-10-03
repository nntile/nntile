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

#ifdef NNTILE_USE_CUDA

template<typename T>
void run_cuda(Index nelems, Scalar alpha, const std::vector<T> &src,
        Scalar beta, std::vector<T> &dst)
{
    // Copy to device
    T *dev_src, *dev_dst;
    cudaError_t cuda_err = cudaMalloc(&dev_src, sizeof(T)*nelems);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMalloc(&dev_dst, sizeof(T)*nelems);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_src, &src[0], sizeof(T)*nelems,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(dev_dst, &dst[0], sizeof(T)*nelems,
            cudaMemcpyHostToDevice);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Init stream
    cudaStream_t stream;
    cuda_err = cudaStreamCreate(&stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Launch low-level CUDA kernel
    add_inplace::cuda<T>(stream, nelems, alpha, dev_src, beta, dev_dst);
    cuda_err = cudaStreamSynchronize(stream);
    TEST_ASSERT(cuda_err == cudaSuccess);
    // Copy result and deallocate device memory
    cuda_err = cudaMemcpy(&dst[0], dev_dst, sizeof(T)*nelems,
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
void validate(Index nelems, int test_index_a, int test_index_b)
{
    using Y = typename T::repr_t;
    const Y eps = 2 * T::epsilon;
    // Init test input
    Scalar alpha = (1.0)/Scalar(test_index_a);
    Scalar beta = (1.0)/Scalar(test_index_b);
    std::vector<T> src(nelems), dst(nelems);
    for(Index i = 0; i < nelems; ++i)
    {
        src[i] = Y(2*i+1-nelems);
        dst[i] = Y(2*nelems-i);
    }
    std::vector<T> dst_save(dst);
    std::cout << "Run kernel::add_inplace::cpu<" << T::short_name << ">\n";
    add_inplace::cpu<T>(nelems, alpha, &src[0], beta, &dst[0]);
    for(Index i = 0; i < nelems; ++i)
    {
        Y val_ref = alpha*Y(2*i+1-nelems) + beta*Y(2*nelems-i);
        TEST_ASSERT(std::abs(Y{dst[i]}-val_ref)/std::abs(val_ref) <= eps);
    }
    std::cout << "OK: kernel::add_inplace::cpu<" << T::short_name << ">\n";
#ifdef NNTILE_USE_CUDA
    // Check low-level CUDA kernel
    dst = dst_save;
    std::cout << "Run kernel::add_inplace::cuda<" << T::short_name << ">\n";
    run_cuda<T>(nelems, alpha, src, beta, dst);
    for(Index i = 0; i < nelems; ++i)
    {
        Y val_ref = alpha*Y(2*i+1-nelems) + beta*Y(2*nelems-i);
        TEST_ASSERT(std::abs(Y{dst[i]}-val_ref)/std::abs(val_ref) <= eps);
    }
    std::cout << "OK: kernel::add_inplace::cuda<" << T::short_name << ">\n";
#endif // NNTILE_USE_CUDA
}

// Struct to hold test data and reference results
template<typename T>
struct TestData
{
    using Y = typename T::repr_t;
    Index nelems; // Number of elements
    Scalar alpha, beta; // Scalar factors

    std::vector<T> src;
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
        Y src_val = static_cast<Y>(data.src[i]);
        Y dst_val = static_cast<Y>(data.dst_ref[i]);
        data.dst_ref[i] = static_cast<T>(data.alpha * src_val + data.beta * dst_val);
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

    data.src.resize(data.nelems);
    data.dst_init.resize(data.nelems);
    data.dst_ref.resize(data.nelems);

    switch(strategy)
    {
        // Non-random input generation
        case DataGen::PRESET:
            for(Index i = 0; i < data.nelems; ++i)
            {
                data.src[i] = Y(2 * i + 1 - data.nelems);
                data.dst_init[i] = Y(2 * data.nelems - i);
            }
            break;
        // Specific random initialization
        case DataGen::RANDOM:
            std::mt19937 gen(42);
            std::uniform_real_distribution<Y> dist(-2.0, 2.0);
            for(Index i = 0; i < data.nelems; ++i)
            {
                data.src[i] = dist(gen);
                data.dst_init[i] = dist(gen);
            }
            break;
    }
}

// Get test data and reference results
template<typename T>
TestData<T> get_test_data(Index nelems, Scalar alpha, Scalar beta, DataGen strategy)
{
    TestData<T> data;
    data.nelems = nelems;
    data.alpha = alpha;
    data.beta = beta;

    // Generate data by a provided strategy
    generate_data(data, strategy);

    // Compute reference outputs
    reference_add_inplace(data);
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
        eps_check = 1e-6;
    }
    else if (std::is_same_v<T, fp64_t>)
    {
        eps_check = 1e-12;
    }
    else
    {
        throw std::runtime_error("Unsupported data type");
    }

    for(Index i = 0; i < data.nelems; ++i)
    {
        Y dst_ref = static_cast<Y>(data.dst_ref[i]);

        REQUIRE_THAT(
            static_cast<Y>(dst_out[i]),
            WithinAbs(dst_ref, eps_check) || WithinRel(dst_ref, eps_check)
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
            "[kernel][add_inplace][cpu][nelems=" +
            std::to_string(data.nelems) +
            "][alpha=" + std::to_string(data.alpha) +
            "][beta=" + std::to_string(data.beta) + "]"
        )
        {
            cpu<T>(data.nelems, data.alpha, &data.src[0], data.beta, &dst_cpu[0]);
        };
    }
    else
    {
        cpu<T>(data.nelems, data.alpha, &data.src[0], data.beta, &dst_cpu[0]);
        verify_results(data, dst_cpu);
    }
}

#ifdef NNTILE_USE_CUDA

// Helper function to run CUDA test and verify results
template<typename T, bool run_bench>
void run_cuda_test(TestData<T>& data)
{
    T *dev_src, *dev_dst;
    CUDA_CHECK(cudaMalloc(&dev_src, sizeof(T) * data.nelems),
               "cudaMalloc dev_src");
    CUDA_CHECK(cudaMalloc(&dev_dst, sizeof(T) * data.nelems),
               "cudaMalloc dev_dst");

    std::vector<T> dst_cuda(data.dst_init);

    CUDA_CHECK(cudaMemcpy(dev_src, &data.src[0], sizeof(T) * data.nelems,
                          cudaMemcpyHostToDevice), "cudaMemcpy dev_src");
    CUDA_CHECK(cudaMemcpy(dev_dst, &dst_cuda[0], sizeof(T) * data.nelems,
                          cudaMemcpyHostToDevice), "cudaMemcpy dev_dst");

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
            cuda<T>(stream, data.nelems, data.alpha, dev_src, data.beta, dev_dst);
            cudaStreamSynchronize(stream);
        };
    }
    else
    {
        cuda<T>(stream, data.nelems, data.alpha, dev_src, data.beta, dev_dst);
        CUDA_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

        CUDA_CHECK(cudaMemcpy(&dst_cuda[0], dev_dst, sizeof(T) * data.nelems,
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
