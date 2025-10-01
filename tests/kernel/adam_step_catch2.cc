/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/adam_step_catch2.cc
 * Fused Adam optimizer step with Catch2 testing framework
 *
 * @version 1.1.0
 * */

 // Corresponding header
#include "nntile/kernel/adam_step.hh"

// Standard libraries
#include <vector>
#include <stdexcept>
#include <limits>
#include <iostream>
#include <cmath>

// Third-party libraries
#include <catch2/catch_all.hpp>

// Use namespaces for shorter code
using namespace Catch;

// Use tested NNTile namespaces
using namespace nntile;
using namespace nntile::kernel;
using namespace nntile::kernel::adam_step;

// Struct to hold test data and reference results
template<typename T>
struct TestData
{
    using Y = typename T::repr_t;
    Index num_elems; // Number of data elements
    Index num_iter; // Iteration number
    Scalar beta_1_s, beta_2_s, eps_s, lr_s, weight_decay_s;
    Y beta_1, beta_2, eps, lr, weight_decay;
    Y eps_check;

    std::vector<T> grad;
    std::vector<T> p_init;
    std::vector<T> first_moment_init;
    std::vector<T> second_moment_init;

    std::vector<T> p_ref;
    std::vector<T> first_moment_ref;
    std::vector<T> second_moment_ref;
};

// Helper function for Data Generation and Reference Implementation
template<typename T>
TestData<T> generate_test_data_and_reference(Index num_elems, Index num_iter)
{
    using Y = typename T::repr_t;
    TestData<T> data;
    data.num_elems = num_elems;
    data.num_iter = num_iter;

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
        data.eps_check = 1e-4;
    }
    else if (std::is_same_v<T, fp64_t>)
    {
        data.eps_check = 1e-8;
    }
    else
    {
        throw std::runtime_error("Unsupported data type");
    }

    data.beta_1_s = GENERATE(0.9, 0.1, 0.0);
    data.beta_2_s = GENERATE(0.999, 0.9, 0.1, 0.0);
    data.eps_s = GENERATE(1e-1, 1e-2, 1e-4, 1e-8);
    data.lr_s = GENERATE(1e-1, 1e-3);
    data.weight_decay_s = GENERATE(0.0, 0.1, 0.2);

    data.beta_1 = data.beta_1_s;
    data.beta_2 = data.beta_2_s;
    data.eps = data.eps_s;
    data.lr = data.lr_s;
    data.weight_decay = data.weight_decay_s;

    data.grad.resize(num_elems);
    data.p_init.resize(num_elems);
    data.first_moment_init.assign(num_elems, T{0.0});
    data.second_moment_init.assign(num_elems, T{0.0});

    for(Index i = 0; i < num_elems; ++i)
    {
        data.grad[i] = Y(2*i+1-num_elems);
        data.p_init[i] = Y(num_elems-i);
        data.first_moment_init[i] = Y(i+1);
        data.second_moment_init[i] = Y(num_elems-i);
    }

    data.p_ref = data.p_init;
    data.first_moment_ref = data.first_moment_init;
    data.second_moment_ref = data.second_moment_init;

    if (num_elems == 0)
    {
        return data;
    }

    const Y alpha = data.lr / (Y{1.0}-std::pow(data.beta_1, num_iter));
    const Y beta = Y{1.0} / std::sqrt(Y{1.0}-std::pow(data.beta_2, num_iter));
    for(Index i = 0; i < num_elems; ++i)
    {
        Y p_val = static_cast<Y>(data.p_ref[i]);
        Y grad_val = static_cast<Y>(data.grad[i]);
        if (data.weight_decay != 0)
        {
            grad_val += data.weight_decay * p_val;
        }
        Y f_val, s_val;
        if(num_iter == 1)
        {
            f_val = (Y{1.0}-data.beta_1) * grad_val;
            s_val = std::sqrt(Y{1.0}-data.beta_2) * std::fabs(grad_val);
        }
        else
        {
            f_val = static_cast<Y>(data.first_moment_ref[i]);
            s_val = static_cast<Y>(data.second_moment_ref[i]);
            f_val = data.beta_1*f_val + (Y{1.0}-data.beta_1)*grad_val;
            s_val = std::hypot(std::sqrt(data.beta_2)*s_val,
                    std::sqrt(Y{1.0}-data.beta_2)*grad_val);
        }
        data.first_moment_ref[i] = static_cast<T>(f_val);
        data.second_moment_ref[i] = static_cast<T>(s_val);
        const Y denom = s_val*beta + data.eps;
        data.p_ref[i] = static_cast<T>(p_val - alpha*f_val/denom);
    }
    return data;
}

// Helper function to verify results
template<typename T>
void verify_results(const TestData<T>& data, const std::vector<T>& p_out,
                    const std::vector<T>& m_out, const std::vector<T>& v_out)
{
    using Y = typename T::repr_t;
    for(Index i = 0; i < data.num_elems; ++i)
    {
        REQUIRE(
            static_cast<Y>(p_out[i]) ==
            Approx(static_cast<Y>(data.p_ref[i])).epsilon(data.eps_check)
        );
        REQUIRE(
            static_cast<Y>(m_out[i]) ==
            Approx(static_cast<Y>(data.first_moment_ref[i])).epsilon(data.eps_check)
        );
        REQUIRE(
            static_cast<Y>(v_out[i]) ==
            Approx(static_cast<Y>(data.second_moment_ref[i])).epsilon(data.eps_check)
        );
    }
}

// Catch2-based tests
TEMPLATE_TEST_CASE(
    "Adam Step Kernel Verification",
    "[adam_step]",
    fp64_t,
    fp32_t,
    fp16_t,
    bf16_t
)
{
    using T = TestType;
    auto data = generate_test_data_and_reference<T>(1024, 5);

    SECTION("CPU Execution", "[cpu]") {
        std::vector<T> p_cpu(data.p_init);
        std::vector<T> first_moment_cpu(data.first_moment_init);
        std::vector<T> second_moment_cpu(data.second_moment_init);

        cpu<T>(
            data.num_iter,
            data.num_elems,
            data.beta_1_s,
            data.beta_2_s,
            data.eps_s,
            data.lr_s,
            data.weight_decay_s,
            &data.grad[0],
            &first_moment_cpu[0],
            &second_moment_cpu[0],
            &p_cpu[0]
        );

        // Verification
        verify_results(data, p_cpu, first_moment_cpu, second_moment_cpu);
    }

#ifdef NNTILE_USE_CUDA
    SECTION("CUDA Execution", "[cuda]") {
        // Copy to device
        T *dev_grad, *dev_first_moment, *dev_second_moment, *dev_p;
        cudaMalloc(&dev_grad, sizeof(T)*data.num_elems);
        cudaMalloc(&dev_first_moment, sizeof(T)*data.num_elems);
        cudaMalloc(&dev_second_moment, sizeof(T)*data.num_elems);
        cudaMalloc(&dev_p, sizeof(T)*data.num_elems);

        std::vector<T> p_cuda(data.p_init);
        std::vector<T> first_moment_cuda(data.first_moment_init);
        std::vector<T> second_moment_cuda(data.second_moment_init);

        cudaMemcpy(dev_grad, &data.grad[0], sizeof(T)*data.num_elems, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_first_moment, &first_moment_cuda[0], sizeof(T)*data.num_elems, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_second_moment, &second_moment_cuda[0], sizeof(T)*data.num_elems, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_p, &p_cuda[0], sizeof(T)*data.num_elems, cudaMemcpyHostToDevice);

        cudaStream_t stream;
        cudaStreamCreate(&stream);

        cuda<T>(stream, data.num_iter, data.num_elems, data.beta_1_s, data.beta_2_s, data.eps_s, data.lr_s, data.weight_decay_s,
            dev_grad, dev_first_moment, dev_second_moment, dev_p);

        cudaStreamSynchronize(stream);

        cudaMemcpy(&first_moment_cuda[0], dev_first_moment, sizeof(T)*data.num_elems, cudaMemcpyDeviceToHost);
        cudaMemcpy(&second_moment_cuda[0], dev_second_moment, sizeof(T)*data.num_elems, cudaMemcpyDeviceToHost);
        cudaMemcpy(&p_cuda[0], dev_p, sizeof(T)*data.num_elems, cudaMemcpyDeviceToHost);

        cudaFree(dev_grad);
        cudaFree(dev_first_moment);
        cudaFree(dev_second_moment);
        cudaFree(dev_p);
        cudaStreamDestroy(stream);

        // Verification
        verify_results(data, p_cuda, first_moment_cuda, second_moment_cuda);
    }
#endif
}
