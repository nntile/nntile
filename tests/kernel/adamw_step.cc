/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/adamw_step.cc
 * Fused AdamW optimizer step
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/kernel/adamw_step.hh"

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
using namespace nntile::kernel::adamw_step;

// Type to acquire reference values
using ref_t = double;

// Struct to hold test data and reference results
template<typename T>
struct TestData
{
    using Y = typename T::repr_t;
    Index num_elems; // Number of data elements
    Index num_iter; // Iteration number
    Scalar beta_1;
    Scalar beta_2;
    Scalar eps;
    Scalar lr;
    Scalar weight_decay;

    Y eps_check;

    std::vector<T> grad;
    std::vector<T> p_init;
    std::vector<T> first_moment_init;
    std::vector<T> second_moment_init;

    std::vector<T> p_ref;
    std::vector<T> first_moment_ref;
    std::vector<T> second_moment_ref;
};

// Reference implementation of the AdamW step
template<typename T>
void reference_adamw_step(TestData<T>& data)
{
    using Y = typename T::repr_t;
    if (data.num_elems == 0)
    {
        return;
    }
    const ref_t beta_1_r = data.beta_1;
    const ref_t beta_2_r = data.beta_2;
    const ref_t lr_r = data.lr;
    const ref_t weight_decay_r = data.weight_decay;
    const ref_t eps_r = data.eps;
    constexpr ref_t one = 1.0;
    const ref_t alpha = lr_r / (one - std::pow(beta_1_r, data.num_iter));
    const ref_t beta = one / std::sqrt(one - std::pow(beta_2_r, data.num_iter));
    for(Index i = 0; i < data.num_elems; ++i)
    {
        ref_t p_val = static_cast<Y>(data.p_init[i]);
        ref_t grad_val = static_cast<Y>(data.grad[i]);
        if (weight_decay_r != 0)
        {
            p_val *= (one - lr_r * weight_decay_r);
        }
        ref_t f_val, s_val;
        if(data.num_iter == 1)
        {
            f_val = (one - beta_1_r) * grad_val;
            s_val = std::sqrt(one - beta_2_r) * std::fabs(grad_val);
        }
        else
        {
            f_val = static_cast<Y>(data.first_moment_init[i]);
            s_val = static_cast<Y>(data.second_moment_init[i]);
            f_val = beta_1_r * f_val + (one - beta_1_r) * grad_val;
            s_val = std::hypot(
                std::sqrt(beta_2_r) * s_val,
                std::sqrt(one - beta_2_r) * grad_val
            );
        }
        data.first_moment_ref[i] = static_cast<T>(static_cast<Y>(f_val));
        data.second_moment_ref[i] = static_cast<T>(static_cast<Y>(s_val));
        const ref_t denom = s_val * beta + eps_r;
        const ref_t p_ref = p_val - alpha * f_val / denom;
        data.p_ref[i] = static_cast<T>(static_cast<Y>(p_ref));
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
    data.first_moment_init.resize(num_elems);
    data.second_moment_init.resize(num_elems);

    data.p_ref.resize(num_elems);
    data.first_moment_ref.resize(num_elems);
    data.second_moment_ref.resize(num_elems);

    switch(strategy)
    {
        // Non-random input generation
        case DataGen::PRESET:
            for(Index i = 0; i < num_elems; ++i)
            {
                data.grad[i] = Y(2 * i + 1 - num_elems);
                data.p_init[i] = Y(num_elems - i);
                data.first_moment_init[i] = Y(i + 1);
                data.second_moment_init[i] = Y(num_elems - i);
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
                data.first_moment_init[i] = 3 * dist(gen);
                data.second_moment_init[i] = std::abs(4 * dist(gen));
            }
    }
}

// Get test input data (reference computation is done separately)
template<typename T>
TestData<T> get_test_input_data(
    Index num_elems,
    Index num_iter,
    Scalar beta_1,
    Scalar beta_2,
    Scalar eps,
    Scalar lr,
    Scalar weight_decay,
    DataGen strategy
)
{
    TestData<T> data;
    // Generate data by a provided strategy
    generate_data(data, num_elems, strategy);
    // Fill in remaining fields of TestData
    data.num_iter = num_iter;
    data.beta_1 = beta_1;
    data.beta_2 = beta_2;
    data.eps = eps;
    data.lr = lr;
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
    const std::vector<T>& m_out,
    const std::vector<T>& v_out
)
{
    using Y = typename T::repr_t;
    for(Index i = 0; i < data.num_elems; ++i)
    {
        Y p_ref = static_cast<Y>(data.p_ref[i]);
        Y m_ref = static_cast<Y>(data.first_moment_ref[i]);
        Y v_ref = static_cast<Y>(data.second_moment_ref[i]);
        REQUIRE_THAT(
            static_cast<Y>(p_out[i]),
            WithinRel(p_ref, data.eps_check)
        );
        REQUIRE_THAT(
            static_cast<Y>(m_out[i]),
            WithinRel(m_ref, data.eps_check)
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
    std::vector<T> first_moment_cpu(data.first_moment_init);
    std::vector<T> second_moment_cpu(data.second_moment_init);

    if constexpr (run_bench)
    {
        BENCHMARK(
            "[kernel][adamw_step][cpu][nelems=" +
            std::to_string(data.num_elems) +
            "][weight_decay=" +
            std::to_string(data.weight_decay) +
            "]"
        )
        {
            cpu<T>(
                data.num_iter,
                data.num_elems,
                data.beta_1,
                data.beta_2,
                data.eps,
                data.lr,
                data.weight_decay,
                &data.grad[0],
                &first_moment_cpu[0],
                &second_moment_cpu[0],
                &p_cpu[0]
            );
        };
    }
    else
    {
        cpu<T>(
            data.num_iter,
            data.num_elems,
            data.beta_1,
            data.beta_2,
            data.eps,
            data.lr,
            data.weight_decay,
            &data.grad[0],
            &first_moment_cpu[0],
            &second_moment_cpu[0],
            &p_cpu[0]
        );
        verify_results(data, p_cpu, first_moment_cpu, second_moment_cpu);
    }
}

#ifdef NNTILE_USE_CUDA

// Helper function to run CUDA test and verify results
template<typename T, bool run_bench>
void run_cuda_test(TestData<T>& data)
{
    T *dev_grad, *dev_first_moment, *dev_second_moment, *dev_p;
    CUDA_CHECK(cudaMalloc(&dev_grad, sizeof(T) * data.num_elems),
               "cudaMalloc dev_grad");
    CUDA_CHECK(cudaMalloc(&dev_first_moment, sizeof(T) * data.num_elems),
               "cudaMalloc dev_first_moment");
    CUDA_CHECK(cudaMalloc(&dev_second_moment, sizeof(T) * data.num_elems),
               "cudaMalloc dev_second_moment");
    CUDA_CHECK(cudaMalloc(&dev_p, sizeof(T) * data.num_elems),
               "cudaMalloc dev_p");

    std::vector<T> p_cuda(data.p_init);
    std::vector<T> first_moment_cuda(data.first_moment_init);
    std::vector<T> second_moment_cuda(data.second_moment_init);

    CUDA_CHECK(cudaMemcpy(dev_grad, &data.grad[0], sizeof(T) * data.num_elems,
                          cudaMemcpyHostToDevice), "cudaMemcpy dev_grad");
    CUDA_CHECK(cudaMemcpy(dev_first_moment, &first_moment_cuda[0],
                          sizeof(T) * data.num_elems, cudaMemcpyHostToDevice),
               "cudaMemcpy dev_first_moment");
    CUDA_CHECK(cudaMemcpy(dev_second_moment, &second_moment_cuda[0],
                          sizeof(T) * data.num_elems, cudaMemcpyHostToDevice),
               "cudaMemcpy dev_second_moment");
    CUDA_CHECK(cudaMemcpy(dev_p, &p_cuda[0], sizeof(T) * data.num_elems,
                          cudaMemcpyHostToDevice), "cudaMemcpy dev_p");

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream), "cudaStreamCreate");

    if constexpr (run_bench)
    {
        BENCHMARK(
            "[kernel][adamw_step][cuda][nelems=" +
            std::to_string(data.num_elems) +
            "][weight_decay=" +
            std::to_string(data.weight_decay) +
            "]"
        )
        {
            cuda<T>(
                stream,
                data.num_iter,
                data.num_elems,
                data.beta_1,
                data.beta_2,
                data.eps,
                data.lr,
                data.weight_decay,
                dev_grad,
                dev_first_moment,
                dev_second_moment,
                dev_p
            );
            cudaStreamSynchronize(stream);
        };
    }
    else
    {
        cuda<T>(
            stream,
            data.num_iter,
            data.num_elems,
            data.beta_1,
            data.beta_2,
            data.eps,
            data.lr,
            data.weight_decay,
            dev_grad,
            dev_first_moment,
            dev_second_moment,
            dev_p
        );
        CUDA_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

        CUDA_CHECK(cudaMemcpy(&first_moment_cuda[0], dev_first_moment,
                              sizeof(T) * data.num_elems, cudaMemcpyDeviceToHost),
                   "cudaMemcpy first_moment_cuda");
        CUDA_CHECK(cudaMemcpy(&second_moment_cuda[0], dev_second_moment,
                              sizeof(T) * data.num_elems, cudaMemcpyDeviceToHost),
                   "cudaMemcpy second_moment_cuda");
        CUDA_CHECK(cudaMemcpy(&p_cuda[0], dev_p, sizeof(T) * data.num_elems,
                              cudaMemcpyDeviceToHost), "cudaMemcpy p_cuda");

        verify_results(data, p_cuda, first_moment_cuda, second_moment_cuda);
    }

    CUDA_CHECK(cudaFree(dev_grad), "cudaFree dev_grad");
    CUDA_CHECK(cudaFree(dev_first_moment), "cudaFree dev_first_moment");
    CUDA_CHECK(cudaFree(dev_second_moment), "cudaFree dev_second_moment");
    CUDA_CHECK(cudaFree(dev_p), "cudaFree dev_p");
    CUDA_CHECK(cudaStreamDestroy(stream), "cudaStreamDestroy");
}
#endif

// Catch2-based tests
TEMPLATE_TEST_CASE(
    "AdamW Step Kernel Verification",
    "[adamw_step]",
    fp64_t,
    fp32_t,
    fp16_t,
    bf16_t
)
{
    using T = TestType;
    const Index num_elems = GENERATE(5, 129);
    const Index num_iter = GENERATE(1, 5);
    const Scalar beta_1 = GENERATE(0.9, 0.1);
    const Scalar beta_2 = GENERATE(0.999, 0.9, 0.1);
    const Scalar eps = GENERATE(1e-1, 1e-2, 1e-4);
    const Scalar lr = GENERATE(1e-1, 1e-3);
    const Scalar weight_decay = GENERATE(0.0, 0.1, 0.2);
    const DataGen strategy = GENERATE(DataGen::PRESET, DataGen::RANDOM);

    auto data = get_test_input_data<T>(
        num_elems,
        num_iter,
        beta_1,
        beta_2,
        eps,
        lr,
        weight_decay,
        strategy
    );

    // Compute reference outputs for verification
    reference_adamw_step(data);

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
    "AdamW Step Kernel Benchmark",
    "[adamw_step][!benchmark]",
    fp64_t,
    fp32_t,
    fp16_t,
    bf16_t
)
{
    using T = TestType;
    const Index num_elems = GENERATE(512, 1024*1024, 4096*16384);
    const Index num_iter = GENERATE(2);
    const Scalar beta_1 = GENERATE(0.9);
    const Scalar beta_2 = GENERATE(0.999);
    const Scalar eps = GENERATE(1e-8);
    const Scalar lr = GENERATE(1e-2);
    const Scalar weight_decay = GENERATE(0.1);
    const DataGen strategy = GENERATE(DataGen::PRESET);

    auto data = get_test_input_data<T>(
        num_elems,
        num_iter,
        beta_1,
        beta_2,
        eps,
        lr,
        weight_decay,
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
