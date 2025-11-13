/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/flash_sdpa_bwd_cudnn.cc
 * Flash Attention Scaled Dot-Product Attention backward pass using cuDNN
 *
 * @version 1.1.0
 */

// Corresponding headers
#include "nntile/kernel/flash_sdpa_bwd_cudnn.hh"

// Standard library headers
#include <vector>
#include <random>
#include <limits>
#include <stdexcept>
#include <cmath>
#include <string>
#include <type_traits>
#include <algorithm>
#include <cstdint>

// Third-party headers
#include <catch2/catch_all.hpp>
#ifdef NNTILE_USE_CUDA
#    include <cudnn.h>
#endif // NNTILE_USE_CUDA

// Other NNTile headers
#include <nntile/kernel/cuda.hh>
#include <nntile/base_types.hh>

using namespace Catch;
using namespace nntile;
using namespace nntile::kernel;
using namespace nntile::kernel::flash_sdpa_bwd_cudnn;

using ref_t = double;

#ifdef NNTILE_USE_CUDA
//! Check cuDNN error and throw runtime_error if error occurs
#    define CUDNN_CHECK(error, message) \
        do { \
            cudnnStatus_t _err = (error); \
            if (_err != CUDNN_STATUS_SUCCESS) \
            { \
                throw std::runtime_error(std::string(message) + ": cuDNN error code " + std::to_string(_err)); \
            } \
        } while (0)
#endif // NNTILE_USE_CUDA

// Enum for data generation strategies
enum class DataGen
{
    PRESET,
    RANDOM
};

// Enum for attention mask types
enum class MaskType
{
    CAUSAL,
    FULL
};

// Struct to hold backward test data
template<typename T>
struct TestData
{
    using Y = typename T::repr_t;
    Index seq;
    Index head;
    Index batch;

    float eps_check;

    std::vector<T> K_init;
    std::vector<T> Q_init;
    std::vector<T> V_init;
    std::vector<T> mask_init;
    std::vector<T> O_init;
    std::vector<fp32_t> lse_init;
    std::vector<T> dO_init;

    std::vector<T> dK_ref;
    std::vector<T> dQ_ref;
    std::vector<T> dV_ref;
};

// Generates deterministic or random inputs together with upstream gradients
template<typename T>
void generate_data(TestData<T> &data,
                   Index seq,
                   Index head,
                   Index batch,
                   DataGen strategy,
                   MaskType mask_type)
{
    using Y = typename T::repr_t;
    data.seq = seq;
    data.head = head;
    data.batch = batch;

    const Index total = batch * seq * head;
    data.K_init.resize(total);
    data.Q_init.resize(total);
    data.V_init.resize(total);
    data.O_init.resize(total);
    data.mask_init.resize(seq * seq);
    data.dO_init.resize(total);
    data.lse_init.resize(batch * seq);

    switch(strategy)
    {
        case DataGen::PRESET:
            for(Index i = 0; i < total; ++i)
            {
                const Y base = Y((i % 23 - 11) * 0.04);
                data.K_init[i] = T(base);
                data.Q_init[i] = T(base * Y(0.7));
                data.V_init[i] = T(base * Y(1.1));
                data.O_init[i] = T(base * Y(-0.3));
                data.dO_init[i] = T(Y(0.02 * ((i % 17) - 8)));
            }
            for(Index i = 0; i < batch * seq; ++i)
            {
                data.lse_init[i] = fp32_t(static_cast<float>((i % 13 - 6) * 0.01f));
            }
            break;
        case DataGen::RANDOM:
        {
            std::mt19937 gen(2024);
            std::uniform_real_distribution<Y> dist_inputs(-0.5, 0.5);
            std::uniform_real_distribution<float> dist_fp32(-1.0f, 1.0f);
            for(Index i = 0; i < total; ++i)
            {
                const Y sample = dist_inputs(gen);
                data.K_init[i] = T(sample);
                data.Q_init[i] = T(dist_inputs(gen));
                data.V_init[i] = T(dist_inputs(gen));
                data.O_init[i] = T(dist_inputs(gen));
                data.dO_init[i] = T(sample * Y(0.5));
            }
            for(Index i = 0; i < batch * seq; ++i)
            {
                data.lse_init[i] = fp32_t(dist_fp32(gen));
            }
            break;
        }
    }

    const T zero_val = T(Y(0));
    const T neg_inf = T(-std::numeric_limits<Y>::infinity());
    switch(mask_type)
    {
        case MaskType::CAUSAL:
            for(Index i = 0; i < seq; ++i)
            {
                for(Index j = 0; j < seq; ++j)
                {
                    const Index idx = i * seq + j;
                    data.mask_init[idx] = (i < j) ? neg_inf : zero_val;
                }
            }
            break;
        case MaskType::FULL:
            std::fill(data.mask_init.begin(), data.mask_init.end(), zero_val);
            break;
    }
}

// reference backward computation for gradients wrt K/Q/V
template<typename T>
void reference_backward(TestData<T> &data)
{
    using Y = typename T::repr_t;
    const ref_t scale = 1.0 / std::sqrt(static_cast<ref_t>(data.head));
    const Index total = data.batch * data.seq * data.head;

    std::vector<ref_t> dK_acc(total, 0.0);
    std::vector<ref_t> dQ_acc(total, 0.0);
    std::vector<ref_t> dV_acc(total, 0.0);

    std::vector<ref_t> scores(data.seq);
    std::vector<ref_t> P(data.seq);
    std::vector<ref_t> dP(data.seq);

    for(Index b = 0; b < data.batch; ++b)
    {
        for(Index i = 0; i < data.seq; ++i)
        {
            for(Index j = 0; j < data.seq; ++j)
            {
                ref_t score = 0.0;
                for(Index h = 0; h < data.head; ++h)
                {
                    const Index q_idx = b * data.seq * data.head + i * data.head + h;
                    const Index k_idx = b * data.seq * data.head + j * data.head + h;
                    score += static_cast<Y>(data.Q_init[q_idx]) * static_cast<Y>(data.K_init[k_idx]);
                }
                score *= scale;
                const Index mask_idx = i * data.seq + j;
                score += static_cast<Y>(data.mask_init[mask_idx]);
                scores[j] = score;
            }

            const Index lse_idx = b * data.seq + i;
            const ref_t lse_val = static_cast<ref_t>(static_cast<float>(data.lse_init[lse_idx]));

            for(Index j = 0; j < data.seq; ++j)
            {
                if(std::isinf(scores[j]) && scores[j] < 0)
                {
                    P[j] = 0.0;
                }
                else
                {
                    P[j] = std::exp(scores[j] - lse_val);
                }
            }

            for(Index j = 0; j < data.seq; ++j)
            {
                ref_t dp = 0.0;
                for(Index h = 0; h < data.head; ++h)
                {
                    const Index o_idx = b * data.seq * data.head + i * data.head + h;
                    const Index v_idx = b * data.seq * data.head + j * data.head + h;
                    dp += static_cast<Y>(data.dO_init[o_idx]) * static_cast<Y>(data.V_init[v_idx]);
                }
                dP[j] = dp;
            }

            ref_t sum_attn_dp = 0.0;
            for(Index h = 0; h < data.head; ++h)
            {
                const Index o_idx = b * data.seq * data.head + i * data.head + h;
                ref_t O_val = static_cast<Y>(data.O_init[o_idx]);
                ref_t dO_val = static_cast<Y>(data.dO_init[o_idx]);
                sum_attn_dp += O_val * dO_val;
            }

            for(Index j = 0; j < data.seq; ++j)
            {
                const ref_t dZ = P[j] * (dP[j] - sum_attn_dp);
                for(Index h = 0; h < data.head; ++h)
                {
                    const Index o_idx = b * data.seq * data.head + i * data.head + h;
                    const Index v_idx = b * data.seq * data.head + j * data.head + h;
                    dV_acc[v_idx] += P[j] * static_cast<Y>(data.dO_init[o_idx]);

                    const Index q_idx = b * data.seq * data.head + i * data.head + h;
                    const Index k_idx = b * data.seq * data.head + j * data.head + h;

                    const ref_t k_val = static_cast<Y>(data.K_init[k_idx]);
                    const ref_t q_val = static_cast<Y>(data.Q_init[q_idx]);

                    dQ_acc[q_idx] += dZ * k_val * scale;
                    dK_acc[k_idx] += dZ * q_val * scale;
                }
            }
        }
    }

    data.dK_ref.resize(total);
    data.dQ_ref.resize(total);
    data.dV_ref.resize(total);

    for(Index idx = 0; idx < total; ++idx)
    {
        const Y k_val = static_cast<Y>(dK_acc[idx]);
        const Y q_val = static_cast<Y>(dQ_acc[idx]);
        const Y v_val = static_cast<Y>(dV_acc[idx]);
        data.dK_ref[idx] = T(k_val);
        data.dQ_ref[idx] = T(q_val);
        data.dV_ref[idx] = T(v_val);
    }
}

// Ensures gradients match reference within tolerance
template<typename T>
void verify_gradients_close(const std::vector<T> &ref_vals,
                            const std::vector<T> &observed,
                            float eps,
                            const char *tensor_name)
{
    using Y = typename T::repr_t;
    ref_t norm = 0.0;
    ref_t diff = 0.0;
    for(size_t idx = 0; idx < ref_vals.size(); ++idx)
    {
        const ref_t ref_val = static_cast<Y>(ref_vals[idx]);
        const ref_t obs_val = static_cast<Y>(observed[idx]);
        norm = std::hypot(norm, ref_val);
        diff = std::hypot(diff, ref_val - obs_val);
    }

    INFO(tensor_name << " gradient mismatch");
    if(norm == 0.0)
    {
        REQUIRE(diff <= eps);
    }
    else
    {
        REQUIRE(diff <= eps * norm);
    }
}

// Helper to create fully configured test data
template<typename T>
TestData<T> get_test_input_data(Index seq,
                                Index head,
                                Index batch,
                                DataGen strategy,
                                MaskType mask_type)
{
    TestData<T> data;
    generate_data(data, seq, head, batch, strategy, mask_type);

    if constexpr(std::is_same_v<T, bf16_t>)
    {
        data.eps_check = 1e-2f;
    }
    else if constexpr(std::is_same_v<T, fp16_t>)
    {
        data.eps_check = 2e-3f;
    }
    else
    {
        data.eps_check = 1e-4f;
    }

    return data;
}

#ifdef NNTILE_USE_CUDA
// Executes the CUDA kernels, validating gradients or benchmarking depending on template flag
template<typename T, bool run_bench>
void run_cuda_test(TestData<T> &data)
{
    const Index total = data.batch * data.seq * data.head;

    T *dev_K = nullptr;
    T *dev_Q = nullptr;
    T *dev_V = nullptr;
    T *dev_O = nullptr;
    T *dev_dO = nullptr;
    T *dev_mask = nullptr;
    T *dev_dK = nullptr;
    T *dev_dQ = nullptr;
    T *dev_dV = nullptr;
    fp32_t *dev_lse = nullptr;
    void *workspace = nullptr;
    std::int64_t workspace_size = 0;

    CUDA_CHECK(cudaMalloc(&dev_K, sizeof(T) * total), "cudaMalloc dev_K");
    CUDA_CHECK(cudaMalloc(&dev_Q, sizeof(T) * total), "cudaMalloc dev_Q");
    CUDA_CHECK(cudaMalloc(&dev_V, sizeof(T) * total), "cudaMalloc dev_V");
    CUDA_CHECK(cudaMalloc(&dev_O, sizeof(T) * total), "cudaMalloc dev_O");
    CUDA_CHECK(cudaMalloc(&dev_dO, sizeof(T) * total), "cudaMalloc dev_dO");
    CUDA_CHECK(cudaMalloc(&dev_mask, sizeof(T) * data.seq * data.seq), "cudaMalloc dev_mask");
    CUDA_CHECK(cudaMalloc(&dev_dK, sizeof(T) * total), "cudaMalloc dev_dK");
    CUDA_CHECK(cudaMalloc(&dev_dQ, sizeof(T) * total), "cudaMalloc dev_dQ");
    CUDA_CHECK(cudaMalloc(&dev_dV, sizeof(T) * total), "cudaMalloc dev_dV");
    CUDA_CHECK(cudaMalloc(&dev_lse, sizeof(fp32_t) * data.batch * data.seq), "cudaMalloc dev_lse");

    CUDA_CHECK(cudaMemcpy(dev_K, data.K_init.data(), sizeof(T) * total, cudaMemcpyHostToDevice),
               "cudaMemcpy dev_K");
    CUDA_CHECK(cudaMemcpy(dev_Q, data.Q_init.data(), sizeof(T) * total, cudaMemcpyHostToDevice),
               "cudaMemcpy dev_Q");
    CUDA_CHECK(cudaMemcpy(dev_V, data.V_init.data(), sizeof(T) * total, cudaMemcpyHostToDevice),
               "cudaMemcpy dev_V");
    CUDA_CHECK(cudaMemcpy(dev_O, data.O_init.data(), sizeof(T) * total, cudaMemcpyHostToDevice),
               "cudaMemcpy dev_O_init");
    CUDA_CHECK(cudaMemcpy(dev_dO, data.dO_init.data(), sizeof(T) * total, cudaMemcpyHostToDevice),
               "cudaMemcpy dev_dO_init");
    CUDA_CHECK(cudaMemcpy(dev_mask, data.mask_init.data(),
                          sizeof(T) * data.seq * data.seq,
                          cudaMemcpyHostToDevice),
               "cudaMemcpy dev_mask");
    CUDA_CHECK(cudaMemcpy(dev_lse, data.lse_init.data(),
                          sizeof(fp32_t) * data.batch * data.seq,
                          cudaMemcpyHostToDevice),
               "cudaMemcpy dev_lse_init");
    CUDA_CHECK(cudaMemset(dev_dK, 0, sizeof(T) * total), "cudaMemset dev_dK");
    CUDA_CHECK(cudaMemset(dev_dQ, 0, sizeof(T) * total), "cudaMemset dev_dQ");
    CUDA_CHECK(cudaMemset(dev_dV, 0, sizeof(T) * total), "cudaMemset dev_dV");

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream), "cudaStreamCreate");

    cudnnHandle_t handle = nullptr;
    CUDNN_CHECK(cudnnCreate(&handle), "cudnnCreate");
    CUDNN_CHECK(cudnnSetStream(handle, stream), "cudnnSetStream");

    auto cleanup = [&]() {
        if(handle != nullptr)
        {
            CUDNN_CHECK(cudnnDestroy(handle), "cudnnDestroy");
            handle = nullptr;
        }
        if(stream != nullptr)
        {
            CUDA_CHECK(cudaStreamDestroy(stream), "cudaStreamDestroy");
            stream = nullptr;
        }

        CUDA_CHECK(cudaFree(dev_K), "cudaFree dev_K");
        CUDA_CHECK(cudaFree(dev_Q), "cudaFree dev_Q");
        CUDA_CHECK(cudaFree(dev_V), "cudaFree dev_V");
        CUDA_CHECK(cudaFree(dev_O), "cudaFree dev_O");
        CUDA_CHECK(cudaFree(dev_dO), "cudaFree dev_dO");
        CUDA_CHECK(cudaFree(dev_mask), "cudaFree dev_mask");
        CUDA_CHECK(cudaFree(dev_dK), "cudaFree dev_dK");
        CUDA_CHECK(cudaFree(dev_dQ), "cudaFree dev_dQ");
        CUDA_CHECK(cudaFree(dev_dV), "cudaFree dev_dV");
        CUDA_CHECK(cudaFree(dev_lse), "cudaFree dev_lse");
        if (workspace != nullptr)
        {
            CUDA_CHECK(cudaFree(workspace), "cudaFree workspace");
            workspace = nullptr;
        }
    };

    auto bwd_graph = prepare_graph<T>(
        handle, data.seq, data.head, data.batch);

    if(bwd_graph == nullptr)
    {
        WARN("cuDNN Flash Attention backward graph preparation failed — skipping test "
             "for this configuration (likely unsupported on this system).");
        cleanup();
        return;
    }

    auto ws_status = bwd_graph->get_workspace_size(workspace_size);
    if(!ws_status.is_good())
    {
        WARN("cuDNN Flash Attention backward workspace query failed — skipping test "
             "for this configuration (likely unsupported on this system).");
        cleanup();
        return;
    }

    if(workspace_size > 0)
    {
        CUDA_CHECK(cudaMalloc(&workspace, static_cast<size_t>(workspace_size)),
                   "cudaMalloc workspace");
    }

    if constexpr(run_bench)
    {
        BENCHMARK(
            "[kernel][flash_sdpa_bwd_cudnn][cuda][seq=" + std::to_string(data.seq) +
            "][head=" + std::to_string(data.head) + "][batch=" + std::to_string(data.batch) + "]")
        {
            kernel::flash_sdpa_bwd_cudnn::execute_graph<T>(
                handle, bwd_graph, dev_K, dev_Q, dev_V, dev_O, dev_dO,
                dev_mask, dev_lse, dev_dK, dev_dQ, dev_dV, workspace);
            cudaStreamSynchronize(stream);
        };
    }
    else
    {
        kernel::flash_sdpa_bwd_cudnn::execute_graph<T>(
            handle, bwd_graph, dev_K, dev_Q, dev_V, dev_O, dev_dO,
            dev_mask, dev_lse, dev_dK, dev_dQ, dev_dV, workspace);
        CUDA_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize backward");

        std::vector<T> dK_host(total);
        std::vector<T> dQ_host(total);
        std::vector<T> dV_host(total);

        CUDA_CHECK(cudaMemcpy(dK_host.data(), dev_dK, sizeof(T) * total, cudaMemcpyDeviceToHost),
                   "cudaMemcpy dK_host");
        CUDA_CHECK(cudaMemcpy(dQ_host.data(), dev_dQ, sizeof(T) * total, cudaMemcpyDeviceToHost),
                   "cudaMemcpy dQ_host");
        CUDA_CHECK(cudaMemcpy(dV_host.data(), dev_dV, sizeof(T) * total, cudaMemcpyDeviceToHost),
                   "cudaMemcpy dV_host");

        verify_gradients_close(data.dK_ref, dK_host, data.eps_check, "dK");
        verify_gradients_close(data.dQ_ref, dQ_host, data.eps_check, "dQ");
        verify_gradients_close(data.dV_ref, dV_host, data.eps_check, "dV");
    }

    bwd_graph.reset();
    cleanup();
}
#endif // NNTILE_USE_CUDA

// Catch2-based tests (only CUDA is supported)
#ifdef NNTILE_USE_CUDA
TEMPLATE_TEST_CASE("Flash SDPA Backward cuDNN Kernel Verification",
                   "[flash_sdpa_bwd_cudnn]",
                   fp16_t,
                   bf16_t)
{
    using T = TestType;
    const Index seq = GENERATE(64);
    const Index head = GENERATE(8, 32);
    const Index batch = GENERATE(1, 4);
    const DataGen strategy = GENERATE(DataGen::PRESET, DataGen::RANDOM);
    const MaskType mask_type = GENERATE(MaskType::FULL, MaskType::CAUSAL);

    auto data = get_test_input_data<T>(seq, head, batch, strategy, mask_type);

    reference_backward(data);

    SECTION("cuda")
    {
        run_cuda_test<T, false>(data);
    }
}

TEMPLATE_TEST_CASE("Flash SDPA Backward cuDNN Kernel Benchmark",
                   "[flash_sdpa_bwd_cudnn][!benchmark]",
                   fp16_t,
                   bf16_t)
{
    using T = TestType;
    const Index seq = GENERATE(128, 1024, 4096);
    const Index head = GENERATE(64, 128);
    const Index batch = GENERATE(4, 8);
    const DataGen strategy = GENERATE(DataGen::RANDOM);
    const MaskType mask_type = GENERATE(MaskType::FULL);

    auto data = get_test_input_data<T>(seq, head, batch, strategy, mask_type);

    SECTION("cuda")
    {
        run_cuda_test<T, true>(data);
    }
}
#endif // NNTILE_USE_CUDA
