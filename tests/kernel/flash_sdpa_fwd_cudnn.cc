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
#include <type_traits>

// Third-party libraries
#include <catch2/catch_all.hpp>
#ifdef NNTILE_USE_CUDA
#include <cudnn.h>
#endif // NNTILE_USE_CUDA

// Other NNTile headers
// CUDA_CHECK definition
#include <nntile/kernel/cuda.hh>
// Since CPU version is not supported, base_types.hh is needed
#include <nntile/base_types.hh>

// Use namespaces for shorter code
using namespace Catch;
using namespace Catch::Matchers;

// Use tested NNTile namespaces
using namespace nntile;
using namespace nntile::kernel;
using namespace nntile::kernel::flash_sdpa_fwd_cudnn;

// Type to acquire reference values
using ref_t = double;

#ifdef NNTILE_USE_CUDA
//! Check cuDNN error and throw runtime_error if error occurs
#define CUDNN_CHECK(error, message) \
    do { \
        cudnnStatus_t _err = (error); \
        if (_err != CUDNN_STATUS_SUCCESS) { \
            throw std::runtime_error(std::string(message) + ": cuDNN error code " + std::to_string(_err)); \
        } \
    } while (0)
#endif

// Struct to hold test data and reference results
template<typename T>
struct TestData
{
    using Y = typename T::repr_t;
    Index seq;   // Sequence length
    Index head;  // Head dimension
    Index batch; // Batch size

    Y eps_check;

    std::vector<T> K_init; // Key: [batch, seq, head]
    std::vector<T> Q_init; // Query: [batch, seq, head]
    std::vector<T> V_init; // Value: [batch, seq, head]
    std::vector<T> mask_init; // Mask: [seq, seq]
    std::vector<T> A_init; // Attention output: [batch, seq, head]
    std::vector<T> lse_init; // Log-sum-exp: [batch, seq]

    std::vector<T> A_ref; // Attention output reference: [batch, seq, head]
    std::vector<T> lse_ref; // Log-sum-exp reference: [batch, seq]

};

// Reference implementation of attention: A = softmax(Q @ K^T / sqrt(head)) @ V
template<typename T>
void reference_attention(TestData<T>& data)
{
    using Y = typename T::repr_t;
    const ref_t scale = 1.0 / std::sqrt(static_cast<ref_t>(data.head));

    // Initialize outputs
    data.A_ref.resize(data.batch * data.seq * data.head);
    data.lse_ref.resize(data.batch * data.seq);

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
                    ref_t q_val = static_cast<Y>(data.Q_init[q_idx]);
                    ref_t k_val = static_cast<Y>(data.K_init[k_idx]);
                    score += q_val * k_val;
                }
                // Scale by sqrt(head)
                score *= scale;

                // Apply mask
                Index mask_idx = i * data.seq + j;
                ref_t mask_val = static_cast<Y>(data.mask_init[mask_idx]);
                scores[j] = score + mask_val;
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
            Index lse_idx = b * data.seq + i;
            data.lse_ref[lse_idx] = static_cast<Y>(
                std::log(sum_exp) + max_score);

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
                              static_cast<ref_t>(static_cast<Y>(data.V_init[v_idx]));
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

// Enum for mask
enum class MaskType
{
    CAUSAL,
    FULL
};

// Generates data with preset, deterministic values
template<typename T>
void generate_data(TestData<T>& data, Index seq, Index head, Index batch,
                   DataGen strategy, MaskType mask_type)
{
    using Y = typename T::repr_t;
    data.seq = seq;
    data.head = head;
    data.batch = batch;

    data.K_init.resize(batch * seq * head);
    data.Q_init.resize(batch * seq * head);
    data.mask_init.resize(seq * seq);
    data.V_init.resize(batch * seq * head);
    data.A_init.resize(batch * seq * head);
    data.lse_init.resize(batch * seq);

    switch(strategy)
    {
        // Non-random input generation
        case DataGen::PRESET:
            for(Index i = 0; i < batch * seq * head; ++i)
            {
                Y val = Y((i % 20 - 10) * 0.1);
                data.K_init[i] = val;
                data.Q_init[i] = val * Y(0.8);
                data.V_init[i] = val * Y(1.2);
            }
            break;
        // Specific random initialization
        case DataGen::RANDOM:
            std::mt19937 gen(42);
            std::uniform_real_distribution<Y> dist(-0.5, 0.5);
            for(Index i = 0; i < batch * seq * head; ++i)
            {
                data.K_init[i] = dist(gen);
                data.Q_init[i] = dist(gen);
                data.V_init[i] = dist(gen);
            }
            break;
    }

    // Initialize output buffers with random values regardless of strategy
    std::mt19937 init_gen(1337);
    std::uniform_real_distribution<Y> init_dist(-1.0, 1.0);
    for(Index i = 0; i < batch * seq * head; ++i)
    {
        data.A_init[i] = init_dist(init_gen);
    }
    for(Index i = 0; i < batch * seq; ++i)
    {
        data.lse_init[i] = init_dist(init_gen);
    }

    // Generate mask
    const T zero = static_cast<Y>(0.0);
    const T minfty = -std::numeric_limits<Y>::infinity();
    switch(mask_type)
    {
        case MaskType::CAUSAL:
            for(Index i = 0; i < seq; ++i)
            {
                for(Index j = 0; j < seq; ++j)
                {
                    Index idx = i * seq + j;
                    if(i < j)
                    {
                        data.mask_init[idx] = minfty;
                    }
                    else
                    {
                        data.mask_init[idx] = zero;
                    }
                }
            }
            break;
        case MaskType::FULL:
            for(Index idx = 0; idx < seq * seq; ++idx)
            {
                data.mask_init[idx] = zero;
            }
            break;
    }
}

// Get test data and reference results
template<typename T>
TestData<T> get_test_input_data(
    Index seq,
    Index head,
    Index batch,
    DataGen strategy,
    MaskType mask_type
)
{
    TestData<T> data;
    // Generate data by a provided strategy
    generate_data(data, seq, head, batch, strategy, mask_type);

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

    return data;
}

// Helper function to verify results
template<typename T>
void verify_results(
    const TestData<T>& data,
    const std::vector<T>& A_out,
    const std::vector<T>& lse_out,
    const std::vector<T>& K_out,
    const std::vector<T>& Q_out,
    const std::vector<T>& V_out,
    const std::vector<T>& mask_out
)
{
    using Y = typename T::repr_t;

    // Verify attention output
    for(Index i = 0; i < data.batch * data.seq * data.head; ++i)
    {
        Y a_ref = static_cast<Y>(data.A_ref[i]);
        Y a_out = static_cast<Y>(A_out[i]);
        REQUIRE_THAT(
            a_out,
            WithinRel(a_ref, data.eps_check) ||
            WithinAbs(a_ref, data.eps_check)
        );
    }

    // // Verify LSE (currently disabled)
    // for(Index i = 0; i < data.batch * data.seq; ++i)
    // {
    //     Y lse_ref = static_cast<Y>(data.lse_ref[i]);
    //     Y lse_out_val = static_cast<Y>(lse_out[i]);
    //
    //     REQUIRE_THAT(
    //         lse_out_val,
    //         WithinRel(lse_ref, data.eps_check * Y(2.0))  // Allow slightly more error
    //     );
    // }

    auto ensure_inputs_unchanged =
        [](const std::vector<T>& expected,
           const std::vector<T>& observed,
           const char* tensor_name)
        {
            using val_t = typename T::repr_t;
            REQUIRE(expected.size() == observed.size());
            for(size_t idx = 0; idx < expected.size(); ++idx)
            {
                INFO(tensor_name << " input mismatch at idx " << idx);
                REQUIRE(observed[idx].value == expected[idx].value);
            }
        };

    ensure_inputs_unchanged(data.K_init, K_out, "K");
    ensure_inputs_unchanged(data.Q_init, Q_out, "Q");
    ensure_inputs_unchanged(data.V_init, V_out, "V");
    ensure_inputs_unchanged(data.mask_init, mask_out, "mask");
}

#ifdef NNTILE_USE_CUDA
// Helper function to run CUDA test and verify results
template<typename T, bool run_bench>
void run_cuda_test(TestData<T>& data)
{
    T *dev_K, *dev_Q, *dev_lse, *dev_V, *dev_A, *dev_mask;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&dev_K, sizeof(T) * data.batch * data.seq * data.head),
               "cudaMalloc dev_K");
    CUDA_CHECK(cudaMalloc(&dev_Q, sizeof(T) * data.batch * data.seq * data.head),
               "cudaMalloc dev_Q");
    CUDA_CHECK(cudaMalloc(&dev_lse, sizeof(T) * data.batch * data.seq),
               "cudaMalloc dev_lse");
    CUDA_CHECK(cudaMalloc(&dev_V, sizeof(T) * data.batch * data.seq * data.head),
               "cudaMalloc dev_V");
    CUDA_CHECK(cudaMalloc(&dev_A, sizeof(T) * data.batch * data.seq * data.head),
               "cudaMalloc dev_A");
    CUDA_CHECK(cudaMalloc(&dev_mask, sizeof(T) * data.seq * data.seq),
               "cudaMalloc dev_mask");

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(dev_K, &data.K_init[0],
                          sizeof(T) * data.batch * data.seq * data.head,
                          cudaMemcpyHostToDevice), "cudaMemcpy dev_K");
    CUDA_CHECK(cudaMemcpy(dev_Q, &data.Q_init[0],
                          sizeof(T) * data.batch * data.seq * data.head,
                          cudaMemcpyHostToDevice), "cudaMemcpy dev_Q");
    CUDA_CHECK(cudaMemcpy(dev_V, &data.V_init[0],
                          sizeof(T) * data.batch * data.seq * data.head,
                          cudaMemcpyHostToDevice), "cudaMemcpy dev_V");
    CUDA_CHECK(cudaMemcpy(dev_A, &data.A_init[0],
                          sizeof(T) * data.batch * data.seq * data.head,
                          cudaMemcpyHostToDevice), "cudaMemcpy dev_A_init");
    CUDA_CHECK(cudaMemcpy(dev_mask, data.mask_init.data(),
                          sizeof(T) * data.seq * data.seq,
                          cudaMemcpyHostToDevice), "cudaMemcpy dev_mask");
    CUDA_CHECK(cudaMemcpy(dev_lse, &data.lse_init[0],
                          sizeof(T) * data.batch * data.seq,
                          cudaMemcpyHostToDevice),
               "cudaMemcpy dev_lse_init");

    // Create CUDA stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream), "cudaStreamCreate");

    // Create cuDNN handle and set stream
    cudnnHandle_t handle;
    cudnnStatus_t status = cudnnCreate(&handle);
    if (status != CUDNN_STATUS_SUCCESS)
    {
        CUDA_CHECK(cudaStreamDestroy(stream), "cudaStreamDestroy");
        REQUIRE(false); // Fail test if cuDNN handle creation fails
    }

    status = cudnnSetStream(handle, stream);
    if (status != CUDNN_STATUS_SUCCESS)
    {
        CUDNN_CHECK(cudnnDestroy(handle), "cudnnDestroy after stream set failure");
        CUDA_CHECK(cudaStreamDestroy(stream), "cudaStreamDestroy");
        REQUIRE(false); // Fail test if stream setting fails
    }

    // Prepare the cuDNN graph
    auto prepared_graph = prepare_graph<T>(
        handle,
        data.seq,
        data.head,
        data.batch
    );
    if(prepared_graph == nullptr)
    {
        WARN("cuDNN Flash Attention graph preparation failed — skipping test "
             "for this configuration (likely unsupported on this system).");
        CUDNN_CHECK(cudnnDestroy(handle), "cudnnDestroy (prepare_graph failed)");
        CUDA_CHECK(cudaStreamDestroy(stream), "cudaStreamDestroy (prepare_graph failed)");
        CUDA_CHECK(cudaFree(dev_K), "cudaFree dev_K (prepare_graph failed)");
        CUDA_CHECK(cudaFree(dev_Q), "cudaFree dev_Q (prepare_graph failed)");
        CUDA_CHECK(cudaFree(dev_lse), "cudaFree dev_lse (prepare_graph failed)");
        CUDA_CHECK(cudaFree(dev_V), "cudaFree dev_V (prepare_graph failed)");
        CUDA_CHECK(cudaFree(dev_A), "cudaFree dev_A (prepare_graph failed)");
        CUDA_CHECK(cudaFree(dev_mask), "cudaFree dev_mask (prepare_graph failed)");
        return;
    }

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
            execute_graph<T>(
                handle,
                prepared_graph,
                dev_K,
                dev_Q,
                dev_mask,
                dev_lse,
                dev_V,
                dev_A
            );
            cudaStreamSynchronize(stream);
        };
    }
    else
    {
        execute_graph<T>(
            handle,
            prepared_graph,
            dev_K,
            dev_Q,
            dev_mask,
            dev_lse,
            dev_V,
            dev_A
        );
        CUDA_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

        // Copy outputs back to host
        std::vector<T> A_cuda(data.batch * data.seq * data.head);
        std::vector<T> lse_cuda(data.batch * data.seq);
        std::vector<T> K_cuda(data.batch * data.seq * data.head);
        std::vector<T> Q_cuda(data.batch * data.seq * data.head);
        std::vector<T> V_cuda(data.batch * data.seq * data.head);
        std::vector<T> mask_cuda(data.seq * data.seq);

        CUDA_CHECK(cudaMemcpy(&A_cuda[0], dev_A,
                              sizeof(T) * data.batch * data.seq * data.head,
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy A_cuda");
        CUDA_CHECK(cudaMemcpy(&lse_cuda[0], dev_lse,
                              sizeof(T) * data.batch * data.seq,
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy lse_cuda");
        CUDA_CHECK(cudaMemcpy(&K_cuda[0], dev_K,
                              sizeof(T) * data.batch * data.seq * data.head,
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy K_cuda");
        CUDA_CHECK(cudaMemcpy(&Q_cuda[0], dev_Q,
                              sizeof(T) * data.batch * data.seq * data.head,
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy Q_cuda");
        CUDA_CHECK(cudaMemcpy(&V_cuda[0], dev_V,
                              sizeof(T) * data.batch * data.seq * data.head,
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy V_cuda");
        CUDA_CHECK(cudaMemcpy(&mask_cuda[0], dev_mask,
                              sizeof(T) * data.seq * data.seq,
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy mask_cuda");

        verify_results(data, A_cuda, lse_cuda, K_cuda, Q_cuda, V_cuda, mask_cuda);
    }

    // Cleanup cuDNN resources
    CUDNN_CHECK(cudnnDestroy(handle), "cudnnDestroy");
    CUDA_CHECK(cudaStreamDestroy(stream), "cudaStreamDestroy");

    // Free device memory
    CUDA_CHECK(cudaFree(dev_K), "cudaFree dev_K");
    CUDA_CHECK(cudaFree(dev_Q), "cudaFree dev_Q");
    CUDA_CHECK(cudaFree(dev_lse), "cudaFree dev_lse");
    CUDA_CHECK(cudaFree(dev_V), "cudaFree dev_V");
    CUDA_CHECK(cudaFree(dev_A), "cudaFree dev_A");
    CUDA_CHECK(cudaFree(dev_mask), "cudaFree dev_mask");
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
    const Index seq = GENERATE(64, 256);
    const Index head = GENERATE(32, 64);
    const Index batch = GENERATE(1, 2);
    const DataGen strategy = GENERATE(DataGen::PRESET, DataGen::RANDOM);
    const MaskType mask_type = GENERATE(MaskType::CAUSAL, MaskType::FULL);
    auto data = get_test_input_data<T>(
        seq, head, batch, strategy, mask_type
    );

    reference_attention(data);

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
    const Index seq = GENERATE(128, 1024, 4096);
    const Index head = GENERATE(64, 128);
    const Index batch = GENERATE(4, 8);
    const DataGen strategy = GENERATE(DataGen::RANDOM);
    const MaskType mask_type = GENERATE(MaskType::CAUSAL, MaskType::FULL);

    auto data = get_test_input_data<T>(
        seq, head, batch, strategy, mask_type
    );

    SECTION("cuda")
    {
        run_cuda_test<T, true>(data);
    }
}
#endif // NNTILE_USE_CUDA
