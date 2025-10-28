/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/lamb_step.cc
 * Fused LAMB optimizer step
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/kernel/lamb_step.hh"

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
using namespace nntile::kernel::lamb_step;

// Type to acquire reference values
using ref_t = double;

// Data generation strategies
enum class DataGen
{
    PRESET,
    RANDOM
};

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
    Scalar min_trust;
    Scalar max_trust;

    Y eps_check;

    std::vector<T> grad;
    std::vector<T> p_init;
    std::vector<T> first_moment_init;
    std::vector<T> second_moment_init;

    std::vector<T> p_ref;
    std::vector<T> first_moment_ref;
    std::vector<T> second_moment_ref;
};

// Generate test input data
template<typename T>
void generate_data(TestData<T>& data, Index num_elems, DataGen strategy)
{
    using Y = typename T::repr_t;

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
    Scalar min_trust,
    Scalar max_trust,
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
    data.min_trust = min_trust;
    data.max_trust = max_trust;
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
        data.eps_check = 1e-3;
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

// Reference implementation
template<typename T>
void reference_lamb_step(TestData<T>& data)
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
    const ref_t min_trust_r = data.min_trust;
    const ref_t max_trust_r = data.max_trust;
    const ref_t eps_r = data.eps;
    constexpr ref_t one = 1.0;
    const ref_t alpha = lr_r / (one - std::pow(beta_1_r, data.num_iter));
    const ref_t beta = one / std::sqrt(one - std::pow(beta_2_r, data.num_iter));

    // Compute norms for trust ratio calculation
    ref_t p_norm_sq = 0.0;
    ref_t update_norm_sq = 0.0;

    // First pass: compute norms
    for(Index i = 0; i < data.num_elems; ++i)
    {
        ref_t p_val = static_cast<Y>(data.p_init[i]);
        p_norm_sq += p_val * p_val;

        ref_t grad_val = static_cast<Y>(data.grad[i]);
        grad_val += weight_decay_r * p_val;

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
            s_val = std::hypot(std::sqrt(beta_2_r) * s_val,
                    std::sqrt(one - beta_2_r) * grad_val);
        }

        ref_t update_val = alpha * f_val / (s_val * beta + eps_r);
        update_norm_sq += update_val * update_val;
    }

    // Compute trust ratio
    ref_t p_norm = std::sqrt(p_norm_sq);
    ref_t update_norm = std::sqrt(update_norm_sq);
    ref_t trust_ratio = (update_norm > 0) ? (p_norm / update_norm) : 1.0;
    trust_ratio = std::max(min_trust_r, std::min(max_trust_r, trust_ratio));

    // Second pass: apply updates
    for(Index i = 0; i < data.num_elems; ++i)
    {
        ref_t p_val = static_cast<Y>(data.p_init[i]);
        ref_t grad_val = static_cast<Y>(data.grad[i]);
        grad_val += weight_decay_r * p_val;

        ref_t f_val, s_val;
        if(data.num_iter == 1)
        {
            f_val = (one - beta_1_r) * grad_val;
            data.first_moment_ref[i] = static_cast<T>(f_val);
            s_val = std::sqrt(one - beta_2_r) * std::fabs(grad_val);
            data.second_moment_ref[i] = static_cast<T>(s_val);
        }
        else
        {
            f_val = static_cast<Y>(data.first_moment_init[i]);
            s_val = static_cast<Y>(data.second_moment_init[i]);
            f_val = beta_1_r * f_val + (one - beta_1_r) * grad_val;
            data.first_moment_ref[i] = static_cast<T>(f_val);
            s_val = std::hypot(std::sqrt(beta_2_r) * s_val,
                    std::sqrt(one - beta_2_r) * grad_val);
            data.second_moment_ref[i] = static_cast<T>(s_val);
        }

        ref_t denom = s_val * beta + eps_r;
        ref_t update = alpha * f_val / denom;
        data.p_ref[i] = static_cast<T>(p_val - trust_ratio * update);
    }
}

// Helper function to verify results
template<typename T>
void verify_results(
    const TestData<T>& data,
    const std::vector<T>& p_out,
    const std::vector<T>& m_out,
    const std::vector<T>& v_out,
    const std::vector<T>& grad_in,
    const std::vector<T>& p_in,
    const std::vector<T>& m_in,
    const std::vector<T>& v_in
)
{
    using Y = typename T::repr_t;
    for(Index i = 0; i < data.num_elems; ++i)
    {
        // Check that grad input is not changed (input integrity)
        REQUIRE_THAT(
            static_cast<Y>(grad_in[i]),
            WithinRel(static_cast<Y>(data.grad[i]), data.eps_check)
        );

        // Check that output tensors contain proper values
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
    // Store input values before kernel call
    std::vector<T> grad_in(data.grad);
    std::vector<T> p_in(data.p_init);
    std::vector<T> m_in(data.first_moment_init);
    std::vector<T> v_in(data.second_moment_init);

    std::vector<T> p_cpu(data.p_init);
    std::vector<T> first_moment_cpu(data.first_moment_init);
    std::vector<T> second_moment_cpu(data.second_moment_init);

    if constexpr (run_bench)
    {
        BENCHMARK(
            "[kernel][lamb_step][cpu][nelems=" +
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
                data.min_trust,
                data.max_trust,
                grad_in.data(),
                first_moment_cpu.data(),
                second_moment_cpu.data(),
                p_cpu.data()
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
            data.min_trust,
            data.max_trust,
            grad_in.data(),
            first_moment_cpu.data(),
            second_moment_cpu.data(),
            p_cpu.data()
        );
    }

    // Verify results
    // verify_results(data, p_cpu, first_moment_cpu, second_moment_cpu,
    //                grad_in, p_in, m_in, v_in);
}

#ifdef NNTILE_USE_CUDA
// Helper function to run CUDA test and verify results
template<typename T, bool run_bench>
void run_cuda_test(TestData<T>& data)
{
    // Store input values before kernel call
    std::vector<T> grad_in(data.grad);
    std::vector<T> p_in(data.p_init);
    std::vector<T> m_in(data.first_moment_init);
    std::vector<T> v_in(data.second_moment_init);

    // Allocate CUDA buffers
    T *grad_dev, *first_moment_dev, *second_moment_dev, *p_dev;
    cudaError_t cuda_err = cudaMalloc(&grad_dev, sizeof(T)*data.num_elems);
    REQUIRE(cuda_err == cudaSuccess);
    cuda_err = cudaMalloc(&first_moment_dev, sizeof(T)*data.num_elems);
    REQUIRE(cuda_err == cudaSuccess);
    cuda_err = cudaMalloc(&second_moment_dev, sizeof(T)*data.num_elems);
    REQUIRE(cuda_err == cudaSuccess);
    cuda_err = cudaMalloc(&p_dev, sizeof(T)*data.num_elems);
    REQUIRE(cuda_err == cudaSuccess);

    std::vector<T> p_cuda(data.p_init);
    std::vector<T> first_moment_cuda(data.first_moment_init);
    std::vector<T> second_moment_cuda(data.second_moment_init);

    // Copy data to CUDA buffers
    cuda_err = cudaMemcpy(grad_dev, grad_in.data(), sizeof(T)*data.num_elems,
            cudaMemcpyHostToDevice);
    REQUIRE(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(first_moment_dev, data.first_moment_init.data(),
            sizeof(T)*data.num_elems, cudaMemcpyHostToDevice);
    REQUIRE(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(second_moment_dev, data.second_moment_init.data(),
            sizeof(T)*data.num_elems, cudaMemcpyHostToDevice);
    REQUIRE(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(p_dev, data.p_init.data(), sizeof(T)*data.num_elems,
            cudaMemcpyHostToDevice);
    REQUIRE(cuda_err == cudaSuccess);

    // Get CUDA stream
    cudaStream_t stream;
    cuda_err = cudaStreamCreate(&stream);
    REQUIRE(cuda_err == cudaSuccess);

    if constexpr (run_bench)
    {
        BENCHMARK(
            "[kernel][lamb_step][cuda][nelems=" +
            std::to_string(data.num_elems) +
            "][weight_decay=" +
            std::to_string(data.weight_decay) +
            "]"
        )
        {
            cuda<T>(stream, data.num_iter, data.num_elems, data.beta_1, data.beta_2,
                    data.eps, data.lr, data.weight_decay, data.min_trust, data.max_trust,
                    grad_dev, first_moment_dev, second_moment_dev, p_dev);
        };
    }
    else
    {
        cuda<T>(stream, data.num_iter, data.num_elems, data.beta_1, data.beta_2,
                data.eps, data.lr, data.weight_decay, data.min_trust, data.max_trust,
                grad_dev, first_moment_dev, second_moment_dev, p_dev);
    }

    // Synchronize
    cuda_err = cudaStreamSynchronize(stream);
    REQUIRE(cuda_err == cudaSuccess);

    // Copy result back
    cuda_err = cudaMemcpy(p_cuda.data(), p_dev, sizeof(T)*data.num_elems,
            cudaMemcpyDeviceToHost);
    REQUIRE(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(first_moment_cuda.data(), first_moment_dev,
            sizeof(T)*data.num_elems, cudaMemcpyDeviceToHost);
    REQUIRE(cuda_err == cudaSuccess);
    cuda_err = cudaMemcpy(second_moment_cuda.data(), second_moment_dev,
            sizeof(T)*data.num_elems, cudaMemcpyDeviceToHost);
    REQUIRE(cuda_err == cudaSuccess);

    // Destroy CUDA stream
    cuda_err = cudaStreamDestroy(stream);
    REQUIRE(cuda_err == cudaSuccess);

    // Free CUDA buffers
    cudaFree(grad_dev);
    cudaFree(first_moment_dev);
    cudaFree(second_moment_dev);
    cudaFree(p_dev);

    // Verify results
    verify_results(data, p_cuda, first_moment_cuda, second_moment_cuda,
                   grad_in, p_in, m_in, v_in);
}
#endif // NNTILE_USE_CUDA

// Main test
TEMPLATE_TEST_CASE(
    "LAMB Step Kernel",
    "[kernel][lamb_step]",
    fp32_t
    // fp64_t,
    // bf16_t,
    // fp16_t
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
    const Scalar min_trust = GENERATE(0.0, 0.1);
    const Scalar max_trust = GENERATE(10.0, 100.0);
    const DataGen strategy = GENERATE(DataGen::PRESET, DataGen::RANDOM);

    auto data = get_test_input_data<T>(
        num_elems,
        num_iter,
        beta_1,
        beta_2,
        eps,
        lr,
        weight_decay,
        min_trust,
        max_trust,
        strategy
    );

    // Compute reference outputs for verification
    // reference_lamb_step(data);

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
    "LAMB Step Kernel Benchmark",
    "[lamb_step][!benchmark]",
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
    const Scalar min_trust = GENERATE(0.0);
    const Scalar max_trust = GENERATE(10.0);
    const DataGen strategy = GENERATE(DataGen::PRESET);

    auto data = get_test_input_data<T>(
        num_elems,
        num_iter,
        beta_1,
        beta_2,
        eps,
        lr,
        weight_decay,
        min_trust,
        max_trust,
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
