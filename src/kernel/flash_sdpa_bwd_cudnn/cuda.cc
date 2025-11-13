/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/flash_sdpa_bwd_cudnn/cuda.cc
 * Flash attention scaled dot-product attention backward pass using cuDNN
 * Frontend API
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/flash_sdpa_bwd_cudnn/cuda.hh"
#include <cudnn_frontend.h>
#ifdef NNTILE_USE_CUDA
#include <cudnn.h>
#endif
#include <cuda_runtime.h>
#include <stdexcept>
#include <cmath>
#include <unordered_map>
#include <type_traits>
#include <iostream>

namespace fe = cudnn_frontend;

namespace nntile::kernel::flash_sdpa_bwd_cudnn
{

// Tensor UIDs for the graph (use standard int64_t for cuDNN frontend)
constexpr ::int64_t Q_UID = 1;
constexpr ::int64_t K_UID = 2;
constexpr ::int64_t V_UID = 3;
constexpr ::int64_t O_UID = 4;
constexpr ::int64_t MASK_UID = 5;
constexpr ::int64_t STATS_UID = 6;
constexpr ::int64_t DO_UID = 7;
constexpr ::int64_t DQ_UID = 8;
constexpr ::int64_t DK_UID = 9;
constexpr ::int64_t DV_UID = 10;

namespace
{

template<typename T>
inline fe::DataType_t get_data_type()
{
    if constexpr (std::is_same_v<T, fp16_t>) {
        return fe::DataType_t::HALF;
    } else if constexpr (std::is_same_v<T, bf16_t>) {
        return fe::DataType_t::BFLOAT16;
    } else {
        static_assert(std::is_same_v<T, fp16_t> || std::is_same_v<T, bf16_t>,
                      "Unsupported data type for flash_sdpa_bwd_cudnn");
    }
}

} // namespace

template<typename T>
FlashSdpaGraph prepare_graph(cudnnHandle_t handle, Index seq, Index head,
                             Index batch)
    noexcept
//! Prepare cuDNN graph for flash attention backward pass
{
    try {
        fe::DataType_t data_type = get_data_type<T>();

        FlashSdpaGraph graph = std::make_shared<fe::graph::Graph>();
        graph->set_io_data_type(data_type)
            .set_intermediate_data_type(fe::DataType_t::FLOAT)
            .set_compute_data_type(fe::DataType_t::FLOAT);

        ::int64_t b = batch;
        ::int64_t num_heads = 1;
        ::int64_t s = seq;
        ::int64_t d = head;
        float attn_scale = 1.0f / std::sqrt(static_cast<float>(d));

        auto make_io_tensor =
            [graph, data_type, b, num_heads, s, d](const char *name,
                                                   ::int64_t uid,
                                                   bool output = false) {
                auto tensor = graph->tensor(
                    fe::graph::Tensor_attributes()
                        .set_name(name)
                        .set_uid(uid)
                        .set_dim({b, num_heads, s, d})
                        .set_stride({num_heads * s * d, s * d, d, 1})
                        .set_data_type(data_type));
                if (output) {
                    tensor->set_output(true);
                }
                return tensor;
            };

        auto Q_tensor = make_io_tensor("Q", Q_UID);
        auto K_tensor = make_io_tensor("K", K_UID);
        auto V_tensor = make_io_tensor("V", V_UID);
        auto O_tensor = make_io_tensor("O", O_UID);
        auto dO_tensor = make_io_tensor("dO", DO_UID);

        auto Mask_tensor = graph->tensor(fe::graph::Tensor_attributes()
                                             .set_name("Bias")
                                             .set_uid(MASK_UID)
                                             .set_dim({1, 1, s, s})
                                             .set_stride({s * s, s * s, s, 1})
                                             .set_data_type(data_type));

        auto Stats_tensor = graph->tensor(fe::graph::Tensor_attributes()
                                              .set_name("Stats")
                                              .set_uid(STATS_UID)
                                              .set_dim({b, num_heads, s, 1})
                                              .set_stride({num_heads * s, s, 1, 1})
                                              .set_data_type(fe::DataType_t::FLOAT));

        auto bwd_options = fe::graph::SDPA_backward_attributes()
                               .set_name("flash_attention_bwd")
                               .set_attn_scale(attn_scale)
                               .set_bias(Mask_tensor)
                               .set_causal_mask(false);

        auto bwd_outputs = graph->sdpa_backward(Q_tensor,
                                               K_tensor,
                                               V_tensor,
                                               O_tensor,
                                               dO_tensor,
                                               Stats_tensor,
                                               bwd_options);

        auto dQ_tensor = bwd_outputs[0];
        auto dK_tensor = bwd_outputs[1];
        auto dV_tensor = bwd_outputs[2];

        dQ_tensor->set_output(true)
            .set_dim({b, num_heads, s, d})
            .set_stride({num_heads * s * d, s * d, d, 1})
            .set_uid(DQ_UID);
        dK_tensor->set_output(true)
            .set_dim({b, num_heads, s, d})
            .set_stride({num_heads * s * d, s * d, d, 1})
            .set_uid(DK_UID);
        dV_tensor->set_output(true)
            .set_dim({b, num_heads, s, d})
            .set_stride({num_heads * s * d, s * d, d, 1})
            .set_uid(DV_UID);

        auto build_status = graph->build(handle, {fe::HeurMode_t::A});
        if (!build_status.is_good()) {
            std::cerr << "cuDNN backward graph build failed!" << std::endl;
            std::cerr << "Build status code: "
                      << static_cast<int>(build_status.get_code())
                      << std::endl;
            std::cerr << "Build status message: "
                      << build_status.get_message() << std::endl;
            std::cerr << "Dimensions: seq=" << seq
                      << ", head=" << head
                      << ", batch=" << batch << std::endl;
            std::cerr << "Data type: "
                      << (std::is_same_v<T, fp16_t> ? "fp16" : "bf16")
                      << std::endl;
            return nullptr;
        }

        return graph;
    } catch (...) {
        return nullptr;
    }
}

template<typename T>
void execute_graph(cudnnHandle_t handle, const FlashSdpaGraph &prepared_graph,
                   const T *K, const T *Q, const T *V, const T *O,
                   const T *dO, const T *mask, const fp32_t *logsumexp,
                   T *dK, T *dQ, T *dV, void *workspace)
    noexcept
//! Execute prepared cuDNN graph for flash attention backward pass
{
    std::unordered_map<::int64_t, void*> variant_pack = {
        {Q_UID, const_cast<T*>(Q)},
        {K_UID, const_cast<T*>(K)},
        {V_UID, const_cast<T*>(V)},
        {O_UID, const_cast<T*>(O)},
        {DO_UID, const_cast<T*>(dO)},
        {MASK_UID, const_cast<T*>(mask)},
        {STATS_UID, const_cast<fp32_t*>(logsumexp)},
        {DQ_UID, dQ},
        {DK_UID, dK},
        {DV_UID, dV}
    };

    auto exec_status = prepared_graph->execute(handle, variant_pack, workspace);
    if(!exec_status.is_good())
    {
        std::cerr << "cuDNN backward graph execution failed!" << std::endl;
        std::cerr << "Exec status code: "
                    << static_cast<int>(exec_status.get_code())
                    << std::endl;
        std::cerr << "Exec status message: "
                    << exec_status.get_message() << std::endl;
    }
    (void)exec_status;
}

// Explicit instantiations
template
FlashSdpaGraph prepare_graph<fp16_t>(cudnnHandle_t handle, Index seq,
                                     Index head, Index batch) noexcept;

template
FlashSdpaGraph prepare_graph<bf16_t>(cudnnHandle_t handle, Index seq,
                                     Index head, Index batch) noexcept;

template
void execute_graph<fp16_t>(cudnnHandle_t handle,
                           const FlashSdpaGraph &prepared_graph,
                           const fp16_t *K,
                           const fp16_t *Q,
                           const fp16_t *V,
                           const fp16_t *O,
                           const fp16_t *dO,
                           const fp16_t *mask,
                           const fp32_t *logsumexp,
                           fp16_t *dK,
                           fp16_t *dQ,
                           fp16_t *dV,
                           void *workspace) noexcept;

template
void execute_graph<bf16_t>(cudnnHandle_t handle,
                           const FlashSdpaGraph &prepared_graph,
                           const bf16_t *K,
                           const bf16_t *Q,
                           const bf16_t *V,
                           const bf16_t *O,
                           const bf16_t *dO,
                           const bf16_t *mask,
                           const fp32_t *logsumexp,
                           bf16_t *dK,
                           bf16_t *dQ,
                           bf16_t *dV,
                           void *workspace) noexcept;

} // namespace nntile::kernel::flash_sdpa_bwd_cudnn
