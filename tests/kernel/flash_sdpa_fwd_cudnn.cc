/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/flash_sdpa_fwd_cudnn.cc
 * Flash Attention Scaled Dot-Product Attention forward pass using cuDNN
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/kernel/flash_sdpa_fwd_cudnn.hh"

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
using namespace nntile::kernel::flash_sdpa_fwd_cudnn;

// Type to acquire reference values
using ref_t = double;

// Struct to hold test data and reference results
template<typename T>
struct TestData
{
    using Y = typename T::repr_t;
    Index seq;   // Sequence length
    Index head;  // Head dimension
    Index batch; // Batch size

    Y eps_check;

    std::vector<T> K;          // Key: [batch, seq, head]
    std::vector<T> Q;          // Query: [batch, seq, head]
    std::vector<int> seq_lengths_q; // Sequence lengths for Q (per batch element)
    std::vector<int> seq_lengths_kv; // Sequence lengths for K/V (per batch element)
    std::vector<int> lo_win_idx;    // Lower window index for causal masking
    std::vector<int> hi_win_idx;    // Upper window index for causal masking
    std::vector<T> V;          // Value: [batch, seq, head]
    std::vector<T> A_ref;      // Attention output reference: [batch, seq, head]
    std::vector<T> logsumexp_ref; // Log-sum-exp reference: [batch, seq]

    // For reference computation, we still need a mask representation
    std::vector<bool> mask_ref; // Reference mask: [batch, seq, seq] (true = keep, false = mask)
};

// Reference implementation of attention: A = softmax(Q @ K^T / sqrt(head)) @ V
template<typename T>
void reference_attention(TestData<T>& data)
{
    using Y = typename T::repr_t;
    const ref_t scale = 1.0 / std::sqrt(static_cast<ref_t>(data.head));

    // Initialize outputs
    data.A_ref.resize(data.batch * data.seq * data.head);
    data.logsumexp_ref.resize(data.batch * data.seq);

    // Loop over batch
    for(Index b = 0; b < data.batch; ++b)
    {
        // Loop over query positions
        for(Index i = 0; i < data.seq; ++i)
        {
            // Compute attention scores: Q[i] @ K^T
            std::vector<ref_t> scores(data.seq);

            for(Index j = 0; j < data.seq; ++j)
            {
                ref_t score = 0.0;
                // Dot product: Q[b, i, :] @ K[b, j, :]
                for(Index h = 0; h < data.head; ++h)
                {
                    Index q_idx = b * data.seq * data.head + i * data.head + h;
                    Index k_idx = b * data.seq * data.head + j * data.head + h;
                    score += static_cast<ref_t>(static_cast<Y>(data.Q[q_idx])) *
                             static_cast<ref_t>(static_cast<Y>(data.K[k_idx]));
                }
                // Scale by sqrt(head)
                score *= scale;

                // Apply mask
                Index mask_idx = b * data.seq * data.seq + i * data.seq + j;
                if(!data.mask_ref[mask_idx])
                {
                    score = -std::numeric_limits<ref_t>::infinity();
                }

                scores[j] = score;
            }

            // Compute softmax: exp(score - max) / sum(exp(score - max))
            ref_t max_score = -std::numeric_limits<ref_t>::infinity();
            for(Index j = 0; j < data.seq; ++j)
            {
                if(!std::isinf(scores[j]) && scores[j] > max_score)
                {
                    max_score = scores[j];
                }
            }

            std::vector<ref_t> exp_scores(data.seq);
            ref_t sum_exp = 0.0;
            for(Index j = 0; j < data.seq; ++j)
            {
                if(std::isinf(scores[j]) && scores[j] < 0)
                {
                    exp_scores[j] = 0.0;
                }
                else
                {
                    exp_scores[j] = std::exp(scores[j] - max_score);
                    sum_exp += exp_scores[j];
                }
            }

            // Store logsumexp: log(sum_exp) + max_score
            Index logsumexp_idx = b * data.seq + i;
            data.logsumexp_ref[logsumexp_idx] =
                static_cast<T>(static_cast<Y>(std::log(sum_exp) + max_score));

            // Normalize to get attention weights
            std::vector<ref_t> attn_weights(data.seq);
            for(Index j = 0; j < data.seq; ++j)
            {
                attn_weights[j] = exp_scores[j] / sum_exp;
            }

            // Compute weighted sum of values: attn_weights @ V
            for(Index h = 0; h < data.head; ++h)
            {
                ref_t output = 0.0;
                for(Index j = 0; j < data.seq; ++j)
                {
                    Index v_idx = b * data.seq * data.head + j * data.head + h;
                    output += attn_weights[j] *
                              static_cast<ref_t>(static_cast<Y>(data.V[v_idx]));
                }
                Index out_idx = b * data.seq * data.head + i * data.head + h;
                data.A_ref[out_idx] = static_cast<T>(static_cast<Y>(output));
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
void generate_data(TestData<T>& data, Index seq, Index head, Index batch,
                   DataGen strategy)
{
    using Y = typename T::repr_t;
    data.seq = seq;
    data.head = head;
    data.batch = batch;

    data.K.resize(batch * seq * head);
    data.Q.resize(batch * seq * head);
    data.mask_ref.resize(batch * seq * seq);
    data.V.resize(batch * seq * head);
    data.seq_lengths_q.resize(batch);
    data.seq_lengths_kv.resize(batch);
    data.lo_win_idx.resize(batch);
    data.hi_win_idx.resize(batch);

    switch(strategy)
    {
        // Non-random input generation
        case DataGen::PRESET:
            for(Index i = 0; i < batch * seq * head; ++i)
            {
                Y val = Y((i % 20 - 10) * 0.1);
                data.K[i] = val;
                data.Q[i] = val * Y(0.8);
                data.V[i] = val * Y(1.2);
            }
            // Initialize sequence lengths (all sequences use full length for now)
            for(Index b = 0; b < batch; ++b)
            {
                data.seq_lengths_q[b] = seq;
                data.seq_lengths_kv[b] = seq;
                // For causal masking: currIdx - loWinIdx to currIdx + hiWinIdx
                data.lo_win_idx[b] = seq; // Look at all previous positions
                data.hi_win_idx[b] = 0;   // Don't look at future positions (causal)
            }
            // Causal mask: upper triangular part is masked
            for(Index b = 0; b < batch; ++b)
            {
                for(Index i = 0; i < seq; ++i)
                {
                    for(Index j = 0; j < seq; ++j)
                    {
                        Index idx = b * seq * seq + i * seq + j;
                        data.mask_ref[idx] = (j <= i); // true = keep, false = mask
                    }
                }
            }
            break;
        // Specific random initialization
        case DataGen::RANDOM:
            std::mt19937 gen(42);
            std::uniform_real_distribution<Y> dist(-0.5, 0.5);
            for(Index i = 0; i < batch * seq * head; ++i)
            {
                data.K[i] = dist(gen);
                data.Q[i] = dist(gen);
                data.V[i] = dist(gen);
            }
            // Initialize sequence lengths (all sequences use full length for now)
            for(Index b = 0; b < batch; ++b)
            {
                data.seq_lengths_q[b] = seq;
                data.seq_lengths_kv[b] = seq;
                // For causal masking
                data.lo_win_idx[b] = seq;
                data.hi_win_idx[b] = 0;
            }
            // Random causal mask
            std::uniform_real_distribution<Y> mask_dist(0.0, 1.0);
            for(Index b = 0; b < batch; ++b)
            {
                for(Index i = 0; i < seq; ++i)
                {
                    for(Index j = 0; j < seq; ++j)
                    {
                        Index idx = b * seq * seq + i * seq + j;
                        if(j > i)
                        {
                            // Causal mask: future positions are masked
                            data.mask_ref[idx] = false;
                        }
                        else
                        {
                            // Randomly mask some positions in the past
                            data.mask_ref[idx] = (mask_dist(gen) > 0.2);
                        }
                    }
                }
            }
            break;
    }
}

// Get test data and reference results
template<typename T>
TestData<T> get_test_data(
    Index seq,
    Index head,
    Index batch,
    DataGen strategy
)
{
    TestData<T> data;
    // Generate data by a provided strategy
    generate_data(data, seq, head, batch, strategy);

    // Set accuracy threshold for each precision
    if (std::is_same_v<T, bf16_t>)
    {
        data.eps_check = 5e-2;  // BF16 has lower precision
    }
    else if (std::is_same_v<T, fp16_t>)
    {
        data.eps_check = 1e-2;  // FP16 has better precision
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

    // Compute reference outputs
    reference_attention(data);
    return data;
}

// Helper function to verify results
template<typename T>
void verify_results(
    const TestData<T>& data,
    const std::vector<T>& A_out,
    const std::vector<T>& logsumexp_out
)
{
    using Y = typename T::repr_t;

    // Verify attention output
    for(Index i = 0; i < data.batch * data.seq * data.head; ++i)
    {
        Y a_ref = static_cast<Y>(data.A_ref[i]);
        Y a_out = static_cast<Y>(A_out[i]);

        // Skip if reference is very small (relative error not meaningful)
        if(std::abs(a_ref) < Y(1e-6))
        {
            REQUIRE(std::abs(a_out) < data.eps_check);
        }
        else
        {
            REQUIRE_THAT(
                a_out,
                WithinRel(a_ref, data.eps_check)
            );
        }
    }

    // Verify logsumexp
    for(Index i = 0; i < data.batch * data.seq; ++i)
    {
        Y lse_ref = static_cast<Y>(data.logsumexp_ref[i]);
        Y lse_out = static_cast<Y>(logsumexp_out[i]);

        REQUIRE_THAT(
            lse_out,
            WithinRel(lse_ref, data.eps_check * Y(2.0))  // Allow slightly more error
        );
    }
}

#ifdef NNTILE_USE_CUDA
// Helper function to run CUDA test and verify results
template<typename T, bool run_bench>
void run_cuda_test(TestData<T>& data)
{
    T *dev_K, *dev_Q, *dev_logsumexp, *dev_V, *dev_A;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&dev_K, sizeof(T) * data.batch * data.seq * data.head),
               "cudaMalloc dev_K");
    CUDA_CHECK(cudaMalloc(&dev_Q, sizeof(T) * data.batch * data.seq * data.head),
               "cudaMalloc dev_Q");
    CUDA_CHECK(cudaMalloc(&dev_logsumexp, sizeof(T) * data.batch * data.seq),
               "cudaMalloc dev_logsumexp");
    CUDA_CHECK(cudaMalloc(&dev_V, sizeof(T) * data.batch * data.seq * data.head),
               "cudaMalloc dev_V");
    CUDA_CHECK(cudaMalloc(&dev_A, sizeof(T) * data.batch * data.seq * data.head),
               "cudaMalloc dev_A");

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(dev_K, &data.K[0],
                          sizeof(T) * data.batch * data.seq * data.head,
                          cudaMemcpyHostToDevice), "cudaMemcpy dev_K");
    CUDA_CHECK(cudaMemcpy(dev_Q, &data.Q[0],
                          sizeof(T) * data.batch * data.seq * data.head,
                          cudaMemcpyHostToDevice), "cudaMemcpy dev_Q");
    CUDA_CHECK(cudaMemcpy(dev_V, &data.V[0],
                          sizeof(T) * data.batch * data.seq * data.head,
                          cudaMemcpyHostToDevice), "cudaMemcpy dev_V");

    // Initialize logsumexp to zero
    CUDA_CHECK(cudaMemset(dev_logsumexp, 0, sizeof(T) * data.batch * data.seq),
               "cudaMemset dev_logsumexp");

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream), "cudaStreamCreate");

    if constexpr (run_bench)
    {
        BENCHMARK(
            "[kernel][flash_sdpa_fwd_cudnn][cuda][seq=" +
            std::to_string(data.seq) +
            "][head=" +
            std::to_string(data.head) +
            "][batch=" +
            std::to_string(data.batch) +
            "]"
        )
        {
            cuda<T>(
                stream,
                data.seq,
                data.head,
                data.batch,
                dev_K,
                dev_Q,
                nullptr,  // mask is not used - cuDNN uses sequence lengths and window indices
                dev_logsumexp,
                dev_V,
                dev_A
            );
            cudaStreamSynchronize(stream);
        };
    }
    else
    {
        cuda<T>(
            stream,
            data.seq,
            data.head,
            data.batch,
            dev_K,
            dev_Q,
            nullptr,  // mask is not used - cuDNN uses sequence lengths and window indices
            dev_logsumexp,
            dev_V,
            dev_A
        );
        CUDA_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

        // Copy outputs back to host
        std::vector<T> A_cuda(data.batch * data.seq * data.head);
        std::vector<T> logsumexp_cuda(data.batch * data.seq);

        CUDA_CHECK(cudaMemcpy(&A_cuda[0], dev_A,
                              sizeof(T) * data.batch * data.seq * data.head,
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy A_cuda");
        CUDA_CHECK(cudaMemcpy(&logsumexp_cuda[0], dev_logsumexp,
                              sizeof(T) * data.batch * data.seq,
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy logsumexp_cuda");

        verify_results(data, A_cuda, logsumexp_cuda);
    }

    // Free device memory
    CUDA_CHECK(cudaFree(dev_K), "cudaFree dev_K");
    CUDA_CHECK(cudaFree(dev_Q), "cudaFree dev_Q");
    CUDA_CHECK(cudaFree(dev_logsumexp), "cudaFree dev_logsumexp");
    CUDA_CHECK(cudaFree(dev_V), "cudaFree dev_V");
    CUDA_CHECK(cudaFree(dev_A), "cudaFree dev_A");
    CUDA_CHECK(cudaStreamDestroy(stream), "cudaStreamDestroy");
}
#endif // NNTILE_USE_CUDA

// Catch2-based tests (only CUDA is supported)
#ifdef NNTILE_USE_CUDA
TEMPLATE_TEST_CASE(
    "Flash SDPA Forward cuDNN Kernel Verification",
    "[flash_sdpa_fwd_cudnn]",
    fp16_t,
    bf16_t
)
{
    using T = TestType;
    const Index seq = GENERATE(4, 8, 16);
    const Index head = GENERATE(16, 32, 64);
    const Index batch = GENERATE(1, 2);
    const DataGen strategy = GENERATE(DataGen::PRESET, DataGen::RANDOM);

    auto data = get_test_data<T>(
        seq, head, batch,
        strategy
    );

    SECTION("cuda")
    {
        run_cuda_test<T, false>(data);
    }
}

// Catch2-based benchmarks
TEMPLATE_TEST_CASE(
    "Flash SDPA Forward cuDNN Kernel Benchmark",
    "[flash_sdpa_fwd_cudnn][!benchmark]",
    fp16_t,
    bf16_t
)
{
    using T = TestType;
    const Index seq = GENERATE(128, 256);
    const Index head = GENERATE(64, 128);
    const Index batch = GENERATE(4, 8);
    const DataGen strategy = GENERATE(DataGen::RANDOM);

    auto data = get_test_data<T>(
        seq, head, batch,
        strategy
    );

    SECTION("cuda")
    {
        run_cuda_test<T, true>(data);
    }
}
#endif // NNTILE_USE_CUDA
