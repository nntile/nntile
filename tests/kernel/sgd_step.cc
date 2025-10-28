/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/sgd_step.cc
 * Fused SGD with momentum optimizer step
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/kernel/sgd_step.hh"

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
using namespace nntile::kernel::sgd_step;

// Type to acquire reference values
using ref_t = double;

// Struct to hold test data and reference results
template<typename T>
struct TestData
{
    using Y = typename T::repr_t;
    Index num_elems; // Number of data elements

    Scalar momentum;
    Scalar lr;
    Scalar weight_decay;
    bool nesterov;

    Y eps_check;

    std::vector<T> grad;
    std::vector<T> p_init;
    std::vector<T> velocity_init;

    std::vector<T> p_ref;
    std::vector<T> velocity_ref;
};

// Reference implementation of the SGD with momentum step
template<typename T>
void reference_sgd_step(TestData<T>& data)
{
    using Y = typename T::repr_t;
    if (data.num_elems == 0)
    {
        return;
    }
    const ref_t momentum_r = data.momentum;
    const ref_t lr_r = data.lr;
    const ref_t weight_decay_r = data.weight_decay;
    for(Index i = 0; i < data.num_elems; ++i)
    {
        ref_t p_val = static_cast<Y>(data.p_init[i]);
        ref_t grad_val = static_cast<Y>(data.grad[i]);
        grad_val += weight_decay_r * p_val;
        ref_t velocity_val = static_cast<Y>(data.velocity_init[i]);
        // velocity = momentum * velocity + lr * grad
        velocity_val = momentum_r * velocity_val + lr_r * grad_val;
        data.velocity_ref[i] = static_cast<T>(static_cast<Y>(velocity_val));
        // Update parameters
        if (data.nesterov)
        {
            // Nesterov: p = p - lr * (grad + momentum * velocity)
            const ref_t effective_grad = grad_val + momentum_r * velocity_val;
            const ref_t p_ref = p_val - lr_r * effective_grad;
            data.p_ref[i] = static_cast<T>(static_cast<Y>(p_ref));
        }
        else
        {
            // Standard momentum: p = p - velocity
            const ref_t p_ref = p_val - velocity_val;
            data.p_ref[i] = static_cast<T>(static_cast<Y>(p_ref));
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
void generate_data(TestData<T>& data, Index num_elems, DataGen strategy)
{
    using Y = typename T::repr_t;
    data.num_elems = num_elems;

    data.grad.resize(num_elems);
    data.p_init.resize(num_elems);
    data.velocity_init.resize(num_elems);

    data.p_ref.resize(num_elems);
    data.velocity_ref.resize(num_elems);

    switch(strategy)
    {
        // Non-random input generation
        case DataGen::PRESET:
            for(Index i = 0; i < num_elems; ++i)
            {
                data.grad[i] = Y(2 * i + 1 - num_elems);
                data.p_init[i] = Y(num_elems - i);
                data.velocity_init[i] = Y(0); // Initialize velocity to 0
            }
            break;
        // Specific random initialization
        case DataGen::RANDOM:
            std::mt19937 gen(42);
            std::uniform_real_distribution<Y> dist(1.0, 2.0);
            for(Index i = 0; i < num_elems; ++i)
            {
                data.grad[i] = dist(gen);
                data.p_init[i] = 2 * dist(gen);
                data.velocity_init[i] = Y(0); // Initialize velocity to 0
            }
    }
}

// Get test input data (reference computation is done separately)
template<typename T>
TestData<T> get_test_input_data(
    Index num_elems,
    Scalar momentum,
    Scalar lr,
    Scalar weight_decay,
    bool nesterov,
    DataGen strategy
)
{
    TestData<T> data;
    // Generate data by a provided strategy
    generate_data(data, num_elems, strategy);
    // Fill in remaining fields of TestData
    data.momentum = momentum;
    data.lr = lr;
    data.weight_decay = weight_decay;
    data.nesterov = nesterov;
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
    const std::vector<T>& v_out
)
{
    using Y = typename T::repr_t;
    for(Index i = 0; i < data.num_elems; ++i)
    {
        Y p_ref = static_cast<Y>(data.p_ref[i]);
        Y v_ref = static_cast<Y>(data.velocity_ref[i]);
        REQUIRE_THAT(
            static_cast<Y>(p_out[i]),
            WithinRel(p_ref, data.eps_check)
        );
        REQUIRE_THAT(
            static_cast<Y>(v_out[i]),
            WithinRel(v_ref, data.eps_check)
        );
    }
}

// Helper function to run CPU test and verify results
template<typename T, bool run_bench>
void run_cpu_test(TestData<T>& data)
{
    std::vector<T> p_cpu(data.p_init);
    std::vector<T> velocity_cpu(data.velocity_init);

    if constexpr (run_bench)
    {
        BENCHMARK(
            "[kernel][sgd_step][cpu][nelems=" +
            std::to_string(data.num_elems) +
            "][weight_decay=" +
            std::to_string(data.weight_decay) +
            "]"
        )
        {
            cpu<T>(
                data.num_elems,
                data.momentum,
                data.lr,
                data.weight_decay,
                data.nesterov,
                &data.grad[0],
                &velocity_cpu[0],
                &p_cpu[0]
            );
        };
    }
    else
    {
        cpu<T>(
            data.num_elems,
            data.momentum,
            data.lr,
            data.weight_decay,
            data.nesterov,
            &data.grad[0],
            &velocity_cpu[0],
            &p_cpu[0]
        );
        verify_results(data, p_cpu, velocity_cpu);
    }
}

#ifdef NNTILE_USE_CUDA

// Helper function to run CUDA test and verify results
template<typename T, bool run_bench>
void run_cuda_test(TestData<T>& data)
{
    T *dev_grad, *dev_velocity, *dev_p;
    CUDA_CHECK(cudaMalloc(&dev_grad, sizeof(T) * data.num_elems),
               "cudaMalloc dev_grad");
    CUDA_CHECK(cudaMalloc(&dev_velocity, sizeof(T) * data.num_elems),
               "cudaMalloc dev_velocity");
    CUDA_CHECK(cudaMalloc(&dev_p, sizeof(T) * data.num_elems),
               "cudaMalloc dev_p");

    std::vector<T> p_cuda(data.p_init);
    std::vector<T> velocity_cuda(data.velocity_init);

    CUDA_CHECK(cudaMemcpy(dev_grad, &data.grad[0], sizeof(T) * data.num_elems,
                          cudaMemcpyHostToDevice), "cudaMemcpy dev_grad");
    CUDA_CHECK(cudaMemcpy(dev_velocity, &velocity_cuda[0],
                          sizeof(T) * data.num_elems, cudaMemcpyHostToDevice),
               "cudaMemcpy dev_velocity");
    CUDA_CHECK(cudaMemcpy(dev_p, &p_cuda[0], sizeof(T) * data.num_elems,
                          cudaMemcpyHostToDevice), "cudaMemcpy dev_p");

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream), "cudaStreamCreate");

    if constexpr (run_bench)
    {
        BENCHMARK(
            "[kernel][sgd_step][cuda][nelems=" +
            std::to_string(data.num_elems) +
            "][weight_decay=" +
            std::to_string(data.weight_decay) +
            "]"
        )
        {
            cuda<T>(
                stream,
                data.num_elems,
                data.momentum,
                data.lr,
                data.weight_decay,
                data.nesterov,
                dev_grad,
                dev_velocity,
                dev_p
            );
            cudaStreamSynchronize(stream);
        };
    }
    else
    {
        cuda<T>(
            stream,
            data.num_elems,
            data.momentum,
            data.lr,
            data.weight_decay,
            data.nesterov,
            dev_grad,
            dev_velocity,
            dev_p
        );
        CUDA_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

        CUDA_CHECK(cudaMemcpy(&velocity_cuda[0], dev_velocity,
                              sizeof(T) * data.num_elems, cudaMemcpyDeviceToHost),
                   "cudaMemcpy velocity_cuda");
        CUDA_CHECK(cudaMemcpy(&p_cuda[0], dev_p, sizeof(T) * data.num_elems,
                              cudaMemcpyDeviceToHost), "cudaMemcpy p_cuda");

        verify_results(data, p_cuda, velocity_cuda);
    }

    CUDA_CHECK(cudaFree(dev_grad), "cudaFree dev_grad");
    CUDA_CHECK(cudaFree(dev_velocity), "cudaFree dev_velocity");
    CUDA_CHECK(cudaFree(dev_p), "cudaFree dev_p");
    CUDA_CHECK(cudaStreamDestroy(stream), "cudaStreamDestroy");
}
#endif

// Catch2-based tests
TEMPLATE_TEST_CASE(
    "SGD Step Kernel Verification",
    "[sgd_step]",
    fp64_t,
    fp32_t,
    fp16_t,
    bf16_t
)
{
    using T = TestType;
    const Index num_elems = GENERATE(5, 129);
    const Scalar momentum = GENERATE(0.9, 0.1, 0.0);
    const Scalar lr = GENERATE(1e-1, 1e-3);
    const Scalar weight_decay = GENERATE(0.0, 0.1, 0.2);
    const bool nesterov = GENERATE(false, true);
    const DataGen strategy = GENERATE(DataGen::PRESET, DataGen::RANDOM);

    auto data = get_test_input_data<T>(
        num_elems,
        momentum,
        lr,
        weight_decay,
        nesterov,
        strategy
    );

    // Compute reference outputs for verification
    reference_sgd_step(data);

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
    "SGD Step Kernel Benchmark",
    "[sgd_step][!benchmark]",
    fp64_t,
    fp32_t,
    fp16_t,
    bf16_t
)
{
    using T = TestType;
    const Index num_elems = GENERATE(512, 1024*1024, 4096*16384);
    const Scalar momentum = GENERATE(0.9);
    const Scalar lr = GENERATE(1e-2);
    const Scalar weight_decay = GENERATE(0.1);
    const bool nesterov = GENERATE(false, true);
    const DataGen strategy = GENERATE(DataGen::PRESET);

    auto data = get_test_input_data<T>(
        num_elems,
        momentum,
        lr,
        weight_decay,
        nesterov,
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
