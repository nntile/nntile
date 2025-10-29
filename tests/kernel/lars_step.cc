/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/lars_step.cc
 * Fused LARS optimizer step
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/kernel/lars_step.hh"

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
// Testing utilities
#include "../../tests/testing.hh"

// Use namespaces for shorter code
using namespace Catch;
using namespace Catch::Matchers;

// Use tested NNTile namespaces
using namespace nntile;
using namespace nntile::kernel;
using namespace nntile::kernel::lars_step;

// Type to acquire reference values
using ref_t = double;

// Struct to hold test data and reference results
template<typename T>
struct TestData
{
    using Y = typename T::repr_t;
    Index num_elems;
    Scalar lr;
    Scalar trust_ratio;
    Scalar weight_norm;
    Scalar grad_norm;
    Scalar weight_decay;

    Y eps_check;

    std::vector<T> grad;
    std::vector<T> p_init;

    std::vector<T> p_ref;
};

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

    data.grad.resize(num_elems);
    data.p_init.resize(num_elems);
    data.p_ref.resize(num_elems);

    switch(strategy)
    {
        // Non-random input generation
        case DataGen::PRESET:
            for(Index i = 0; i < num_elems; ++i)
            {
                data.grad[i] = Y(2 * i + 1 - num_elems);
                data.p_init[i] = Y(num_elems - i);
            }
            break;
        // Random input generation
        case DataGen::RANDOM:
            {
                std::mt19937 gen(42);
                std::uniform_real_distribution<ref_t> dist(-1.0, 1.0);
                for(Index i = 0; i < num_elems; ++i)
                {
                    data.grad[i] = static_cast<T>(dist(gen));
                    data.p_init[i] = static_cast<T>(dist(gen));
                }
            }
            break;
    }
}

// Reference implementation of the LARS step
template<typename T>
void reference_lars_step(TestData<T>& data)
{
    using Y = typename T::repr_t;
    if (data.num_elems == 0)
    {
        return;
    }
    const ref_t lr_r = data.lr;
    const ref_t trust_ratio_r = data.trust_ratio;
    const ref_t weight_norm_r = data.weight_norm;
    const ref_t grad_norm_r = data.grad_norm;
    const ref_t weight_decay_r = data.weight_decay;
    // Cycle over buffers
    for(Index i = 0; i < data.num_elems; ++i)
    {
        ref_t p_val = static_cast<Y>(data.p_init[i]);
        ref_t grad_val = static_cast<Y>(data.grad[i]);
        if (weight_decay_r != 0)
        {
            grad_val += weight_decay_r * p_val;
        }
        // Compute local learning rate
        ref_t local_lr = (grad_norm_r > 0) ? lr_r * weight_norm_r / grad_norm_r : lr_r;
        // Apply trust ratio clipping
        ref_t adapted_lr = std::min(local_lr, lr_r * trust_ratio_r);
        // Update parameters
        ref_t p_new = p_val - adapted_lr * grad_val;
        data.p_ref[i] = static_cast<T>(static_cast<Y>(p_new));
    }
}

// Generates test data with preset parameters
template<typename T>
TestData<T> get_test_input_data(
    Index num_elems,
    Scalar lr,
    Scalar trust_ratio,
    Scalar weight_norm,
    Scalar grad_norm,
    Scalar weight_decay,
    DataGen strategy
)
{
    TestData<T> data;
    // Generate data by a provided strategy
    generate_data(data, num_elems, strategy);
    // Fill in remaining fields of TestData
    data.lr = lr;
    data.trust_ratio = trust_ratio;
    data.weight_norm = weight_norm;
    data.grad_norm = grad_norm;
    data.weight_decay = weight_decay;
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
    const std::vector<T>& p_out,
    const std::vector<T>& grad_original
)
{
    using Y = typename T::repr_t;
    // Verify that gradient has not been modified
    for(Index i = 0; i < data.num_elems; ++i)
    {
        REQUIRE(data.grad[i].value == grad_original[i].value);
    }
    // Verify that output parameters are correct
    for(Index i = 0; i < data.num_elems; ++i)
    {
        Y p_out_val = static_cast<Y>(p_out[i]);
        Y p_ref_val = static_cast<Y>(data.p_ref[i]);
        REQUIRE_THAT(p_out_val, WithinRel(p_ref_val, data.eps_check));
    }
}

// Helper function to run CPU test and verify results
template<typename T, bool run_bench>
void run_cpu_test(TestData<T>& data)
{
    std::vector<T> p_cpu(data.p_init);
    std::vector<T> grad_cpu(data.grad);

    if constexpr (run_bench)
    {
        BENCHMARK(
            "[kernel][lars_step][cpu][nelems=" +
            std::to_string(data.num_elems) +
            "][weight_decay=" +
            std::to_string(data.weight_decay) +
            "]"
        )
        {
            cpu<T>(
                data.num_elems,
                data.lr,
                data.trust_ratio,
                data.weight_norm,
                data.grad_norm,
                data.weight_decay,
                &grad_cpu[0],
                &p_cpu[0]
            );
        };
    }
    else
    {
        cpu<T>(
            data.num_elems,
            data.lr,
            data.trust_ratio,
            data.weight_norm,
            data.grad_norm,
            data.weight_decay,
            &grad_cpu[0],
            &p_cpu[0]
        );
        verify_results(data, p_cpu, grad_cpu);
    }
}

#ifdef NNTILE_USE_CUDA

// Helper function to run CUDA test and verify results
template<typename T, bool run_bench>
void run_cuda_test(TestData<T>& data)
{
    T *dev_grad, *dev_p;
    CUDA_CHECK(cudaMalloc(&dev_grad, sizeof(T) * data.num_elems),
               "cudaMalloc dev_grad");
    CUDA_CHECK(cudaMalloc(&dev_p, sizeof(T) * data.num_elems),
               "cudaMalloc dev_p");

    std::vector<T> p_cuda(data.p_init);
    std::vector<T> grad_cuda(data.grad);

    CUDA_CHECK(cudaMemcpy(dev_grad, &grad_cuda[0], sizeof(T) * data.num_elems,
                          cudaMemcpyHostToDevice), "cudaMemcpy grad -> dev");
    CUDA_CHECK(cudaMemcpy(dev_p, &p_cuda[0], sizeof(T) * data.num_elems,
                          cudaMemcpyHostToDevice), "cudaMemcpy p -> dev");

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream), "cudaStreamCreate");

    if constexpr (run_bench)
    {
        BENCHMARK(
            "[kernel][lars_step][cuda][nelems=" +
            std::to_string(data.num_elems) +
            "][weight_decay=" +
            std::to_string(data.weight_decay) +
            "]"
        )
        {
            cuda<T>(
                stream,
                data.num_elems,
                data.lr,
                data.trust_ratio,
                data.weight_norm,
                data.grad_norm,
                data.weight_decay,
                dev_grad,
                dev_p
            );
        };
    }
    else
    {
        cuda<T>(
            stream,
            data.num_elems,
            data.lr,
            data.trust_ratio,
            data.weight_norm,
            data.grad_norm,
            data.weight_decay,
            dev_grad,
            dev_p
        );
    }

    CUDA_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

    CUDA_CHECK(cudaMemcpy(&p_cuda[0], dev_p, sizeof(T) * data.num_elems,
                          cudaMemcpyDeviceToHost), "cudaMemcpy dev -> p");

    // Copy gradient back to check it wasn't modified
    CUDA_CHECK(cudaMemcpy(&grad_cuda[0], dev_grad, sizeof(T) * data.num_elems,
                          cudaMemcpyDeviceToHost), "cudaMemcpy dev -> grad");

    CUDA_CHECK(cudaStreamDestroy(stream), "cudaStreamDestroy");
    CUDA_CHECK(cudaFree(dev_grad), "cudaFree dev_grad");
    CUDA_CHECK(cudaFree(dev_p), "cudaFree dev_p");

    if constexpr (!run_bench)
    {
        verify_results(data, p_cuda, grad_cuda);
    }
}

#endif // NNTILE_USE_CUDA

// Catch2-based tests
TEMPLATE_TEST_CASE(
    "LARS Step Kernel Verification",
    "[lars_step]",
    fp64_t,
    fp32_t,
    fp16_t,
    bf16_t
)
{
    using T = TestType;
    const Index num_elems = GENERATE(5, 129);
    const Scalar lr = GENERATE(1e-1, 1e-3);
    const Scalar trust_ratio = GENERATE(0.01, 0.02, 0.05);
    const Scalar weight_norm = GENERATE(0.5, 1.0, 2.0);
    const Scalar grad_norm = GENERATE(0.1, 1.0, 10.0);
    const Scalar weight_decay = GENERATE(0.0, 0.1, 0.2);
    const DataGen strategy = GENERATE(DataGen::PRESET, DataGen::RANDOM);

    auto data = get_test_input_data<T>(
        num_elems,
        lr,
        trust_ratio,
        weight_norm,
        grad_norm,
        weight_decay,
        strategy
    );

    // Compute reference outputs for verification
    reference_lars_step(data);

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
    "LARS Step Kernel Benchmark",
    "[lars_step][!benchmark]",
    fp64_t,
    fp32_t,
    fp16_t,
    bf16_t
)
{
    using T = TestType;
    const Index num_elems = GENERATE(5, 129);
    const Scalar lr = GENERATE(1e-1, 1e-3);
    const Scalar trust_ratio = GENERATE(0.01, 0.02, 0.05);
    const Scalar weight_norm = GENERATE(0.5, 1.0, 2.0);
    const Scalar grad_norm = GENERATE(0.1, 1.0, 10.0);
    const Scalar weight_decay = GENERATE(0.0, 0.1, 0.2);
    const DataGen strategy = GENERATE(DataGen::PRESET);

    auto data = get_test_input_data<T>(
        num_elems,
        lr,
        trust_ratio,
        weight_norm,
        grad_norm,
        weight_decay,
        strategy
    );

    // Compute reference outputs for verification
    reference_lars_step(data);

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
