/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/flash_attention.cc
 * Flash attention forward pass kernel tests
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/kernel/flash_attention.hh"

// NNTile CUDA utilities
#ifdef NNTILE_USE_CUDA
#include "nntile/kernel/cuda.hh"
#endif

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

// Use namespaces for shorter code
using namespace Catch;

// Use tested NNTile namespaces
using namespace nntile;
using namespace nntile::kernel;
using namespace nntile::kernel::flash_attention;

// Type to acquire reference values
using ref_t = double;

// Struct to hold test data and reference results
template<typename T>
struct TestData
{
    using Y = typename T::repr_t;
    Index batch;
    Index num_heads;
    Index seq_len;
    Index head_dim;
    Scalar scale;
    Scalar eps_check;

    std::vector<T> Q;
    std::vector<T> K;
    std::vector<T> V;
    std::vector<T> O_ref;
};

// Enum for data generation strategies
enum class DataGen
{
    PRESET,
    RANDOM,
    IDENTITY
};

// Reference implementation using double precision
template<typename T>
void reference_flash_attention(TestData<T>& data)
{
    using Y = typename T::repr_t;
    
    // Iterate over batch and heads
    for(Index b = 0; b < data.batch; ++b)
    {
        for(Index h = 0; h < data.num_heads; ++h)
        {
            // Base offset for current batch and head
            Index base_offset = (b * data.num_heads + h) * data.seq_len * data.head_dim;
            
            // For each query position
            for(Index i = 0; i < data.seq_len; ++i)
            {
                // Compute attention scores: Q[i] @ K^T / scale
                std::vector<ref_t> scores(data.seq_len);
                ref_t max_score = -std::numeric_limits<ref_t>::infinity();
                
                for(Index j = 0; j < data.seq_len; ++j)
                {
                    ref_t score = 0.0;
                    // Dot product between Q[i] and K[j]
                    for(Index d = 0; d < data.head_dim; ++d)
                    {
                        Index q_idx = base_offset + i * data.head_dim + d;
                        Index k_idx = base_offset + j * data.head_dim + d;
                        ref_t q_val = static_cast<Y>(data.Q[q_idx]);
                        ref_t k_val = static_cast<Y>(data.K[k_idx]);
                        score += q_val * k_val;
                    }
                    score = score * data.scale;
                    scores[j] = score;
                    max_score = std::max(max_score, score);
                }
                
                // Apply softmax: compute exp and sum
                ref_t sum_exp = 0.0;
                for(Index j = 0; j < data.seq_len; ++j)
                {
                    scores[j] = std::exp(scores[j] - max_score);
                    sum_exp += scores[j];
                }
                
                // Normalize
                for(Index j = 0; j < data.seq_len; ++j)
                {
                    scores[j] = scores[j] / sum_exp;
                }
                
                // Compute output: O[i] = sum_j(scores[j] * V[j])
                for(Index d = 0; d < data.head_dim; ++d)
                {
                    ref_t output_val = 0.0;
                    for(Index j = 0; j < data.seq_len; ++j)
                    {
                        Index v_idx = base_offset + j * data.head_dim + d;
                        ref_t v_val = static_cast<Y>(data.V[v_idx]);
                        output_val += scores[j] * v_val;
                    }
                    Index o_idx = base_offset + i * data.head_dim + d;
                    data.O_ref[o_idx] = static_cast<T>(static_cast<Y>(output_val));
                }
            }
        }
    }
}

// Generates data with preset or random values
template<typename T>
void generate_data(TestData<T>& data, Index batch, Index num_heads,
        Index seq_len, Index head_dim, DataGen strategy)
{
    using Y = typename T::repr_t;
    
    data.batch = batch;
    data.num_heads = num_heads;
    data.seq_len = seq_len;
    data.head_dim = head_dim;
    data.scale = 1.0 / std::sqrt(static_cast<float>(head_dim));
    
    Index total_size = batch * num_heads * seq_len * head_dim;
    data.Q.resize(total_size);
    data.K.resize(total_size);
    data.V.resize(total_size);
    data.O_ref.resize(total_size);
    
    switch(strategy)
    {
        // Non-random input generation
        case DataGen::PRESET:
            for(Index i = 0; i < total_size; ++i)
            {
                data.Q[i] = Y(0.1 * (i % 10));
                data.K[i] = Y(0.1 * ((i + 1) % 10));
                data.V[i] = Y(0.2 * ((i + 2) % 10));
            }
            break;
        
        // Identity attention (Q = K leads to uniform attention over all positions)
        case DataGen::IDENTITY:
            for(Index i = 0; i < total_size; ++i)
            {
                Y val = Y(1.0);
                data.Q[i] = val;
                data.K[i] = val;
                data.V[i] = Y(0.5 * (i % seq_len));
            }
            break;
        
        // Random initialization
        case DataGen::RANDOM:
            std::mt19937 gen(42);
            std::uniform_real_distribution<Y> dist(-1.0, 1.0);
            for(Index i = 0; i < total_size; ++i)
            {
                data.Q[i] = dist(gen);
                data.K[i] = dist(gen);
                data.V[i] = dist(gen);
            }
            break;
    }
}

// Get test data and reference results
template<typename T>
TestData<T> get_test_data(Index batch, Index num_heads, Index seq_len,
        Index head_dim, DataGen strategy)
{
    TestData<T> data;
    
    // Generate data by a provided strategy
    generate_data(data, batch, num_heads, seq_len, head_dim, strategy);
    
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
        data.eps_check = 1e-4;
    }
    else if (std::is_same_v<T, fp64_t>)
    {
        data.eps_check = 1e-10;
    }
    else
    {
        throw std::runtime_error("Unsupported data type");
    }
    
    // Compute reference outputs
    reference_flash_attention(data);
    
    return data;
}

// Helper function to verify results
template<typename T>
void verify_results(const TestData<T>& data, const std::vector<T>& O_out)
{
    using Y = typename T::repr_t;
    Index total_size = data.batch * data.num_heads * data.seq_len * data.head_dim;
    
    for(Index i = 0; i < total_size; ++i)
    {
        Y ref_val = static_cast<Y>(data.O_ref[i]);
        Y out_val = static_cast<Y>(O_out[i]);
        auto approx = Approx(ref_val).epsilon(data.eps_check);
        REQUIRE(out_val == approx);
    }
}

// Helper function to run CPU test and verify results
template<typename T, bool run_bench>
void run_cpu_test(TestData<T>& data)
{
    std::vector<T> O_cpu(data.O_ref.size());
    
    if constexpr (run_bench)
    {
        BENCHMARK(
            "[kernel][flash_attention][cpu][batch=" +
            std::to_string(data.batch) +
            "][heads=" +
            std::to_string(data.num_heads) +
            "][seq=" +
            std::to_string(data.seq_len) +
            "][dim=" +
            std::to_string(data.head_dim) +
            "]"
        )
        {
            cpu<T>(
                data.batch,
                data.num_heads,
                data.seq_len,
                data.head_dim,
                &data.Q[0],
                &data.K[0],
                &data.V[0],
                data.scale,
                &O_cpu[0]
            );
        };
    }
    else
    {
        cpu<T>(
            data.batch,
            data.num_heads,
            data.seq_len,
            data.head_dim,
            &data.Q[0],
            &data.K[0],
            &data.V[0],
            data.scale,
            &O_cpu[0]
        );
        verify_results(data, O_cpu);
    }
}

#ifdef NNTILE_USE_CUDA

// Helper function to run CUDA test and verify results
template<typename T, bool run_bench>
void run_cuda_test(TestData<T>& data)
{
    Index total_size = data.batch * data.num_heads * data.seq_len * data.head_dim;
    
    T *dev_Q, *dev_K, *dev_V, *dev_O;
    CUDA_CHECK(cudaMalloc(&dev_Q, sizeof(T) * total_size), "cudaMalloc dev_Q");
    CUDA_CHECK(cudaMalloc(&dev_K, sizeof(T) * total_size), "cudaMalloc dev_K");
    CUDA_CHECK(cudaMalloc(&dev_V, sizeof(T) * total_size), "cudaMalloc dev_V");
    CUDA_CHECK(cudaMalloc(&dev_O, sizeof(T) * total_size), "cudaMalloc dev_O");
    
    std::vector<T> O_cuda(total_size);
    
    CUDA_CHECK(cudaMemcpy(dev_Q, &data.Q[0], sizeof(T) * total_size,
                          cudaMemcpyHostToDevice), "cudaMemcpy dev_Q");
    CUDA_CHECK(cudaMemcpy(dev_K, &data.K[0], sizeof(T) * total_size,
                          cudaMemcpyHostToDevice), "cudaMemcpy dev_K");
    CUDA_CHECK(cudaMemcpy(dev_V, &data.V[0], sizeof(T) * total_size,
                          cudaMemcpyHostToDevice), "cudaMemcpy dev_V");
    
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream), "cudaStreamCreate");
    
    if constexpr (run_bench)
    {
        BENCHMARK(
            "[kernel][flash_attention][cuda][batch=" +
            std::to_string(data.batch) +
            "][heads=" +
            std::to_string(data.num_heads) +
            "][seq=" +
            std::to_string(data.seq_len) +
            "][dim=" +
            std::to_string(data.head_dim) +
            "]"
        )
        {
            cuda<T>(
                stream,
                data.batch,
                data.num_heads,
                data.seq_len,
                data.head_dim,
                dev_Q,
                dev_K,
                dev_V,
                data.scale,
                dev_O
            );
            cudaStreamSynchronize(stream);
        };
    }
    else
    {
        cuda<T>(
            stream,
            data.batch,
            data.num_heads,
            data.seq_len,
            data.head_dim,
            dev_Q,
            dev_K,
            dev_V,
            data.scale,
            dev_O
        );
        CUDA_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
        
        CUDA_CHECK(cudaMemcpy(&O_cuda[0], dev_O, sizeof(T) * total_size,
                              cudaMemcpyDeviceToHost), "cudaMemcpy O_cuda");
        
        verify_results(data, O_cuda);
    }
    
    CUDA_CHECK(cudaFree(dev_Q), "cudaFree dev_Q");
    CUDA_CHECK(cudaFree(dev_K), "cudaFree dev_K");
    CUDA_CHECK(cudaFree(dev_V), "cudaFree dev_V");
    CUDA_CHECK(cudaFree(dev_O), "cudaFree dev_O");
    CUDA_CHECK(cudaStreamDestroy(stream), "cudaStreamDestroy");
}
#endif

// Catch2-based tests
TEMPLATE_TEST_CASE(
    "Flash Attention Kernel Verification",
    "[flash_attention]",
    fp64_t,
    fp32_t,
    fp16_t,
    bf16_t
)
{
    using T = TestType;
    const Index batch = GENERATE(1, 2);
    const Index num_heads = GENERATE(1, 2);
    const Index seq_len = GENERATE(4, 8);
    const Index head_dim = GENERATE(4, 8);
    const DataGen strategy = GENERATE(DataGen::PRESET, DataGen::RANDOM);
    
    auto data = get_test_data<T>(batch, num_heads, seq_len, head_dim, strategy);
    
    SECTION("cpu")
    {
        run_cpu_test<T, false>(data);
    }
    
#ifdef NNTILE_USE_CUDA
    SECTION("cuda")
    {
        // Note: CUDA implementation is a placeholder and won't produce correct results
        // until cuDNN integration is completed
        // run_cuda_test<T, false>(data);
        SKIP("CUDA implementation requires cuDNN integration");
    }
#endif
}

// Catch2-based benchmarks
TEMPLATE_TEST_CASE(
    "Flash Attention Kernel Benchmark",
    "[flash_attention][!benchmark]",
    fp64_t,
    fp32_t,
    fp16_t,
    bf16_t
)
{
    using T = TestType;
    const Index batch = GENERATE(2);
    const Index num_heads = GENERATE(8);
    const Index seq_len = GENERATE(64, 128);
    const Index head_dim = GENERATE(64);
    const DataGen strategy = GENERATE(DataGen::RANDOM);
    
    auto data = get_test_data<T>(batch, num_heads, seq_len, head_dim, strategy);
    
    SECTION("cpu")
    {
        run_cpu_test<T, true>(data);
    }
    
#ifdef NNTILE_USE_CUDA
    SECTION("cuda")
    {
        // run_cuda_test<T, true>(data);
        SKIP("CUDA implementation requires cuDNN integration");
    }
#endif
}
