/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/starpu/flash_attention.cc
 * Flash attention StarPU wrapper tests
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/starpu/flash_attention.hh"

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

// NNTile headers
#include "nntile/starpu/config.hh"

// Use namespaces for shorter code
using namespace Catch;

// Use tested NNTile namespaces
using namespace nntile;
using namespace nntile::starpu;

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

// Get test data and reference results
template<typename T>
TestData<T> get_test_data(Index batch, Index num_heads, Index seq_len,
        Index head_dim)
{
    using Y = typename T::repr_t;
    TestData<T> data;
    
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
    
    // Generate random data
    std::mt19937 gen(42);
    std::uniform_real_distribution<Y> dist(-1.0, 1.0);
    for(Index i = 0; i < total_size; ++i)
    {
        data.Q[i] = dist(gen);
        data.K[i] = dist(gen);
        data.V[i] = dist(gen);
    }
    
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

// Catch2-based tests
TEMPLATE_TEST_CASE(
    "Flash Attention StarPU Verification",
    "[flash_attention][starpu]",
    fp64_t,
    fp32_t,
    fp16_t,
    bf16_t
)
{
    using T = TestType;
    
    // Initialize StarPU
    Config starpu_config(1, 0, 0);
    
    const Index batch = GENERATE(1, 2);
    const Index num_heads = GENERATE(1, 2);
    const Index seq_len = GENERATE(4, 8);
    const Index head_dim = GENERATE(4, 8);
    
    auto data = get_test_data<T>(batch, num_heads, seq_len, head_dim);
    
    Index total_size = batch * num_heads * seq_len * head_dim;
    
    // Register data with StarPU
    Handle Q_handle, K_handle, V_handle, O_handle;
    Q_handle.acquire(STARPU_W);
    Q_handle.own(reinterpret_cast<void*>(&data.Q[0]), sizeof(T) * total_size);
    K_handle.acquire(STARPU_W);
    K_handle.own(reinterpret_cast<void*>(&data.K[0]), sizeof(T) * total_size);
    V_handle.acquire(STARPU_W);
    V_handle.own(reinterpret_cast<void*>(&data.V[0]), sizeof(T) * total_size);
    
    std::vector<T> O_out(total_size);
    O_handle.acquire(STARPU_W);
    O_handle.own(reinterpret_cast<void*>(&O_out[0]), sizeof(T) * total_size);
    
    // Submit task
    flash_attention.template get<std::tuple<T>>().submit(
        batch,
        num_heads,
        seq_len,
        head_dim,
        data.scale,
        Q_handle,
        K_handle,
        V_handle,
        O_handle
    );
    
    // Wait for completion
    O_handle.acquire(STARPU_R);
    
    // Verify results
    verify_results(data, O_out);
    
    // Clean up
    O_handle.release();
    Q_handle.release();
    K_handle.release();
    V_handle.release();
}
