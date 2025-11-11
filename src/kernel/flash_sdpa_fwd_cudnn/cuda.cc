/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/flash_sdpa_fwd_cudnn/cuda.cc
 * Flash attention scaled dot-product attention forward pass using cuDNN Frontend API
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/flash_sdpa_fwd_cudnn/cuda.hh"
#include <cudnn_frontend.h>
#ifdef NNTILE_USE_CUDA
#include <cudnn.h>
#endif
#include <cuda_runtime.h>
#include <stdexcept>
#include <cmath>
#include <unordered_map>

namespace fe = cudnn_frontend;

namespace nntile::kernel::flash_sdpa_fwd_cudnn
{

// Tensor UIDs for the graph (use standard int64_t for cuDNN frontend)
constexpr ::int64_t Q_UID = 1;
constexpr ::int64_t K_UID = 2;
constexpr ::int64_t V_UID = 3;
constexpr ::int64_t O_UID = 4;
constexpr ::int64_t MASK_UID = 5;
constexpr ::int64_t STATS_UID = 6;

template<typename T>
FlashSdpaGraph<T>* prepare_graph(cudnnHandle_t handle, Index seq, Index head,
                                   Index batch)
    noexcept
//! Prepare cuDNN graph for flash attention
/*!
 * Prepares and builds the cuDNN graph for flash attention with fixed dimensions.
 * The prepared graph can be reused for multiple executions with different data.
 *
 * @param[in] handle: cuDNN handle (with stream already set)
 * @param[in] seq: Sequence length
 * @param[in] head: Head dimension (d_qk = d_v)
 * @param[in] batch: Batch size
 * @return Pointer to prepared graph structure, or nullptr on error
 * */
{
    try {
        // Determine data type
        fe::DataType_t data_type;
        if constexpr (std::is_same_v<T, fp16_t>) {
            data_type = fe::DataType_t::HALF;
        } else if constexpr (std::is_same_v<T, bf16_t>) {
            data_type = fe::DataType_t::BFLOAT16;
        } else {
            // Unsupported type
            return nullptr;
        }

        // Create result structure
        auto* result = new FlashSdpaGraph<T>();
        result->batch = static_cast<::int64_t>(batch);
        result->seq = static_cast<::int64_t>(seq);
        result->head = static_cast<::int64_t>(head);
        result->has_mask = true;

        // Create a graph
        result->graph = std::make_shared<fe::graph::Graph>();
        result->graph->set_io_data_type(data_type)
            .set_intermediate_data_type(fe::DataType_t::FLOAT)
            .set_compute_data_type(fe::DataType_t::FLOAT);

        // Parameters for SDPA
        ::int64_t b = result->batch;
        ::int64_t num_heads = 1;  // Number of attention heads
        ::int64_t s = result->seq;
        ::int64_t d = result->head;  // Head dimension (d_qk = d_v)
        float attn_scale = 1.0f / std::sqrt(static_cast<float>(d));

        // Define input tensors - cuDNN Flash Attention expects [batch, seq, num_heads, head_dim]
        auto Q_tensor = result->graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("Q")
                                   .set_uid(Q_UID)
                                   .set_dim({b, num_heads, s, d})
                                   .set_stride({num_heads * s * d, s * d, d, 1}));

        auto K_tensor = result->graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("K")
                                   .set_uid(K_UID)
                                   .set_dim({b, num_heads, s, d})
                                   .set_stride({num_heads * s * d, s * d, d, 1}));

        auto V_tensor = result->graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("V")
                                   .set_uid(V_UID)
                                   .set_dim({b, num_heads, s, d})
                                   .set_stride({num_heads * s * d, s * d, d, 1}));

        // Create SDPA options - always use bias (mask)
        auto sdpa_options = fe::graph::SDPA_attributes()
                                .set_name("flash_attention")
                                .set_is_inference(false)  // We want statistics
                                .set_attn_scale(attn_scale);

        auto Mask_tensor = result->graph->tensor(fe::graph::Tensor_attributes()
                                    .set_name("Bias")
                                    .set_uid(MASK_UID)
                                    .set_dim({1, 1, s, s})  // cuDNN expects [1, 1, seq_q, seq_k]
                                    .set_stride({s * s, s * s, s, 1}));

        sdpa_options.set_bias(Mask_tensor);
        sdpa_options.set_causal_mask(false);

        // Execute SDPA
        auto sdpa_output = result->graph->sdpa(Q_tensor, K_tensor, V_tensor, sdpa_options);
        auto O_tensor = sdpa_output[0];
        auto Stats_tensor = sdpa_output[1];

        // Set output properties
        O_tensor->set_output(true)
            .set_dim({b, num_heads, s, d})
            .set_stride({num_heads * s * d, s * d, d, 1})
            .set_uid(O_UID);

        Stats_tensor->set_output(true)
            .set_data_type(fe::DataType_t::FLOAT)
            .set_uid(STATS_UID);

        // Build the graph
        auto build_status = result->graph->build(handle, {fe::HeurMode_t::A});
        if (!build_status.is_good()) {
            std::cerr << "cuDNN graph build failed!" << std::endl;
            std::cerr << "Build status code: " << static_cast<int>(build_status.get_code()) << std::endl;
            std::cerr << "Build status message: " << build_status.get_message() << std::endl;
            std::cerr << "Dimensions: seq=" << seq << ", head=" << head << ", batch=" << batch << std::endl;
            std::cerr << "Data type: " << (std::is_same_v<T, fp16_t> ? "fp16" : "bf16") << std::endl;
            delete result;
            return nullptr;
        }

        // Get workspace size
        auto ws_status = result->graph->get_workspace_size(result->workspace_size);
        if (!ws_status.is_good()) {
            std::cerr << "Failed to get workspace size!" << std::endl;
            std::cerr << "Workspace status code: " << static_cast<int>(ws_status.get_code()) << std::endl;
            std::cerr << "Workspace status message: " << ws_status.get_message() << std::endl;
            delete result;
            return nullptr;
        }

        return result;

    } catch (...) {
        return nullptr;
    }
}

template<typename T>
void execute_graph(cudnnHandle_t handle, const FlashSdpaGraph<T>* prepared_graph,
                   const T* K, const T* Q, const T* mask, T* logsumexp,
                   const T* V, T* A)
    noexcept
//! Execute prepared cuDNN graph for flash attention
/*!
 * Executes a previously prepared cuDNN graph with provided data pointers.
 * The mask tensor is virtually reshaped from [seq, seq] to [1, 1, seq, seq] for cuDNN.
 *
 * @param[in] handle: cuDNN handle (with stream already set)
 * @param[in] prepared_graph: Previously prepared graph structure
 * @param[in] K: Key tensor [batch, seq, head]
 * @param[in] Q: Query tensor [batch, seq, head]
 * @param[in] mask: Mask tensor [seq, seq] (virtually reshaped to [1, 1, seq, seq] for cuDNN)
 * @param[out] logsumexp: Log-sum-exp statistics [batch, seq]
 * @param[in] V: Value tensor [batch, seq, head]
 * @param[out] A: Attention output tensor [batch, seq, head]
 * */
{
    if (prepared_graph == nullptr || prepared_graph->graph == nullptr) {
        return;
    }

    try {
        ::int64_t b = prepared_graph->batch;
        ::int64_t num_heads = 1;
        ::int64_t s = prepared_graph->seq;
        ::int64_t d = prepared_graph->head;

        // Allocate workspace
        void* workspace = nullptr;
        if (prepared_graph->workspace_size > 0) {
            cudaMalloc(&workspace, prepared_graph->workspace_size);
            if (workspace == nullptr) {
                return;
            }
        }

        // Allocate float buffer for stats
        float* stats_float = nullptr;
        cudaMalloc(&stats_float, sizeof(float) * b * num_heads * s);
        if (stats_float == nullptr) {
            if (workspace != nullptr) {
                cudaFree(workspace);
            }
            return;
        }

        // Create variant pack
        std::unordered_map<::int64_t, void*> variant_pack = {
            {Q_UID, const_cast<T*>(Q)},
            {K_UID, const_cast<T*>(K)},
            {V_UID, const_cast<T*>(V)},
            {O_UID, A},
            {MASK_UID, const_cast<T*>(mask)},
            {STATS_UID, stats_float}
        };

        // Execute the graph
        auto exec_status = prepared_graph->graph->execute(handle, variant_pack, workspace);
        if (!exec_status.is_good()) {
            if (workspace != nullptr) {
                cudaFree(workspace);
            }
            cudaFree(stats_float);
            return;
        }

        // Convert float stats to T* logsumexp
        if (logsumexp != nullptr) {
            // WORKAROUND: Copy through host for type conversion
            // TODO: Replace with device kernel for better performance
            std::vector<float> host_stats(b * num_heads * s);
            cudaMemcpy(host_stats.data(), stats_float, sizeof(float) * b * num_heads * s, cudaMemcpyDeviceToHost);
            std::vector<T> host_logsumexp(b * s);
            for (::int64_t i = 0; i < b * s; ++i) {
                // Sum over heads (only 1 head in our case)
                host_logsumexp[i] = static_cast<T>(host_stats[i]);
            }
            cudaMemcpy(logsumexp, host_logsumexp.data(), sizeof(T) * b * s, cudaMemcpyHostToDevice);
        }

        // Cleanup
        if (workspace != nullptr) {
            cudaFree(workspace);
        }
        cudaFree(stats_float);

    } catch (...) {
        return;
    }
}

template<typename T>
void destroy_graph(FlashSdpaGraph<T>* prepared_graph)
    noexcept
//! Destroy prepared cuDNN graph
/*!
 * Frees resources associated with a prepared graph.
 *
 * @param[in] prepared_graph: Graph structure to destroy
 * */
{
    if (prepared_graph != nullptr) {
        delete prepared_graph;
    }
}

template<typename T>
void cuda(cudnnHandle_t handle, Index seq, Index head, Index batch,
          const T* K, const T* Q, const T* mask, T* logsumexp,
          const T* V, T* A)
    noexcept
//! Flash attention forward pass using cuDNN Frontend API (convenience wrapper)
/*!
 * Performs scaled dot-product attention using cuDNN's Flash Attention implementation.
 * This is a convenience function that prepares the graph, executes it, and destroys it.
 * For better performance with repeated calls, use prepare_graph/execute_graph/destroy_graph directly.
 * The mask tensor is virtually reshaped from [batch, seq, seq] to [batch, 1, seq, seq] for cuDNN.
 *
 * @param[in] handle: cuDNN handle (with stream already set)
 * @param[in] seq: Sequence length
 * @param[in] head: Head dimension (d_qk = d_v)
 * @param[in] batch: Batch size
 * @param[in] K: Key tensor [batch, seq, head]
 * @param[in] Q: Query tensor [batch, seq, head]
 * @param[in] mask: Mask tensor [batch, seq, seq] (virtually reshaped to [batch, 1, seq, seq] for cuDNN)
 * @param[out] logsumexp: Log-sum-exp statistics [batch, seq]
 * @param[in] V: Value tensor [batch, seq, head]
 * @param[out] A: Attention output tensor [batch, seq, head]
 * */
{
    // Prepare graph
    auto* prepared_graph = prepare_graph<T>(handle, seq, head, batch);
    if (prepared_graph == nullptr) {
        return;
    }

    // Execute graph
    execute_graph<T>(handle, prepared_graph, K, Q, mask, logsumexp, V, A);

    // Destroy graph
    destroy_graph<T>(prepared_graph);
}

// Explicit instantiations for supported types

// prepare_graph
template
FlashSdpaGraph<fp16_t>* prepare_graph<fp16_t>(cudnnHandle_t handle, Index seq,
                                                Index head, Index batch)
    noexcept;

template
FlashSdpaGraph<bf16_t>* prepare_graph<bf16_t>(cudnnHandle_t handle, Index seq,
                                                Index head, Index batch)
    noexcept;

// execute_graph
template
void execute_graph<fp16_t>(cudnnHandle_t handle, const FlashSdpaGraph<fp16_t>* prepared_graph,
                           const fp16_t* K, const fp16_t* Q, const fp16_t* mask,
                           fp16_t* logsumexp, const fp16_t* V, fp16_t* A)
    noexcept;

template
void execute_graph<bf16_t>(cudnnHandle_t handle, const FlashSdpaGraph<bf16_t>* prepared_graph,
                           const bf16_t* K, const bf16_t* Q, const bf16_t* mask,
                           bf16_t* logsumexp, const bf16_t* V, bf16_t* A)
    noexcept;

// destroy_graph
template
void destroy_graph<fp16_t>(FlashSdpaGraph<fp16_t>* prepared_graph)
    noexcept;

template
void destroy_graph<bf16_t>(FlashSdpaGraph<bf16_t>* prepared_graph)
    noexcept;

// cuda (convenience wrapper)
template
void cuda<fp16_t>(cudnnHandle_t handle, Index seq, Index head, Index batch,
                  const fp16_t* K, const fp16_t* Q, const fp16_t* mask,
                  fp16_t* logsumexp, const fp16_t* V, fp16_t* A)
    noexcept;

template
void cuda<bf16_t>(cudnnHandle_t handle, Index seq, Index head, Index batch,
                  const bf16_t* K, const bf16_t* Q, const bf16_t* mask,
                  bf16_t* logsumexp, const bf16_t* V, bf16_t* A)
    noexcept;

} // namespace nntile::kernel::flash_sdpa_fwd_cudnn
