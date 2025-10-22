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
constexpr ::int64_t STATS_UID = 5;

template<typename T>
void cuda(cudaStream_t stream, Index seq, Index head, Index batch,
          const T* K, const T* Q, const T* mask, T* logsumexp,
          const T* V, T* A)
    noexcept
//! Flash attention forward pass using cuDNN Frontend API
/*!
 * Performs scaled dot-product attention using cuDNN's Flash Attention implementation
 *
 * @param[in] stream: CUDA stream for execution
 * @param[in] seq: Sequence length
 * @param[in] head: Head dimension (d_qk = d_v)
 * @param[in] batch: Batch size
 * @param[in] K: Key tensor [batch, 1, seq, head] (single head)
 * @param[in] Q: Query tensor [batch, 1, seq, head] (single head)
 * @param[in] mask: Not used (masking via cuDNN attributes)
 * @param[out] logsumexp: Log-sum-exp statistics [batch, 1, seq, 1]
 * @param[in] V: Value tensor [batch, 1, seq, head] (single head)
 * @param[out] A: Attention output tensor [batch, 1, seq, head] (single head)
 * */
{
    try {
        // Create cuDNN handle
        cudnnHandle_t handle;
        cudnnStatus_t status = cudnnCreate(&handle);
        if (status != CUDNN_STATUS_SUCCESS) {
            return; // Handle error silently in noexcept function
        }

        // Set stream
        status = cudnnSetStream(handle, stream);
        if (status != CUDNN_STATUS_SUCCESS) {
            cudnnDestroy(handle);
            return;
        }

        // Determine data type
        fe::DataType_t data_type;
        if constexpr (std::is_same_v<T, fp16_t>) {
            data_type = fe::DataType_t::HALF;
        } else if constexpr (std::is_same_v<T, bf16_t>) {
            data_type = fe::DataType_t::BFLOAT16;
        } else if constexpr (std::is_same_v<T, fp32_t>) {
            data_type = fe::DataType_t::FLOAT;
        } else {
            // Unsupported type
            cudnnDestroy(handle);
            return;
        }

        // Create a graph
        auto graph = std::make_shared<fe::graph::Graph>();
        graph->set_io_data_type(data_type)
            .set_intermediate_data_type(fe::DataType_t::FLOAT)
            .set_compute_data_type(fe::DataType_t::FLOAT);

        // Parameters for SDPA (use standard int64_t for cuDNN frontend)
        // Index is already long int, just cast it
        ::int64_t b = static_cast<::int64_t>(batch);
        ::int64_t h = 1;  // Number of heads (we use 1 head with larger dimension)
        ::int64_t s = static_cast<::int64_t>(seq);
        ::int64_t d = static_cast<::int64_t>(head);  // Head dimension
        float attn_scale = 1.0f / std::sqrt(static_cast<float>(head));

        // Define input tensors
        // Input layout from test: [batch, seq, head_dim] where head_dim is what we call 'head'
        // cuDNN expects: [batch, heads, seq, head_dim] where heads=1
        // Stride for [batch, seq, head_dim]: {seq*d, d, 1}
        // Stride for cuDNN [batch, heads=1, seq, head_dim]: {1*s*d, s*d, d, 1} = {s*d, s*d, d, 1}

        auto Q_tensor = graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("Q")
                                   .set_uid(Q_UID)
                                   .set_dim({b, h, s, d})
                                   .set_stride({s * d, s * d, d, 1}));

        auto K_tensor = graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("K")
                                   .set_uid(K_UID)
                                   .set_dim({b, h, s, d})
                                   .set_stride({s * d, s * d, d, 1}));

        auto V_tensor = graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("V")
                                   .set_uid(V_UID)
                                   .set_dim({b, h, s, d})
                                   .set_stride({s * d, s * d, d, 1}));

        // Create SDPA options
        auto sdpa_options = fe::graph::SDPA_attributes()
                                .set_name("flash_attention")
                                .set_is_inference(false)  // We want statistics
                                .set_attn_scale(attn_scale)
                                .set_causal_mask(true);  // Causal mask by default

        // Execute SDPA
        auto [O_tensor, Stats_tensor] = graph->sdpa(Q_tensor, K_tensor, V_tensor, sdpa_options);

        // Set output properties
        // Output layout should match input: [batch, seq, head_dim]
        O_tensor->set_output(true)
            .set_dim({b, h, s, d})
            .set_stride({s * d, s * d, d, 1})
            .set_uid(O_UID);

        Stats_tensor->set_output(true)
            .set_data_type(fe::DataType_t::FLOAT)
            .set_uid(STATS_UID);

        // Build the graph
        auto build_status = graph->build(handle, {fe::HeurMode_t::A});
        if (!build_status.is_good()) {
            cudnnDestroy(handle);
            return;
        }

        // Get workspace size
        ::int64_t workspace_size;
        auto ws_status = graph->get_workspace_size(workspace_size);
        if (!ws_status.is_good()) {
            cudnnDestroy(handle);
            return;
        }

        // Allocate workspace
        void* workspace = nullptr;
        if (workspace_size > 0) {
            cudaMalloc(&workspace, workspace_size);
            if (workspace == nullptr) {
                cudnnDestroy(handle);
                return;
            }
        }

        // Create variant pack (map of tensor UIDs to device pointers)
        std::unordered_map<::int64_t, void*> variant_pack = {
            {Q_UID, const_cast<T*>(Q)},
            {K_UID, const_cast<T*>(K)},
            {V_UID, const_cast<T*>(V)},
            {O_UID, A},
            {STATS_UID, logsumexp}
        };

        // Execute the graph
        auto exec_status = graph->execute(handle, variant_pack, workspace);
        if (!exec_status.is_good()) {
            if (workspace != nullptr) {
                cudaFree(workspace);
            }
            cudnnDestroy(handle);
            return;
        }

        // Cleanup
        if (workspace != nullptr) {
            cudaFree(workspace);
        }
        cudnnDestroy(handle);

    } catch (...) {
        // Catch any exceptions and return silently (noexcept function)
        return;
    }
}

// Explicit instantiation for supported types
template
void cuda<fp16_t>(cudaStream_t stream, Index seq, Index head, Index batch,
                  const fp16_t* K, const fp16_t* Q, const fp16_t* mask,
                  fp16_t* logsumexp, const fp16_t* V, fp16_t* A)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index seq, Index head, Index batch,
                  const bf16_t* K, const bf16_t* Q, const bf16_t* mask,
                  bf16_t* logsumexp, const bf16_t* V, bf16_t* A)
    noexcept;

} // namespace nntile::kernel::flash_sdpa_fwd_cudnn
