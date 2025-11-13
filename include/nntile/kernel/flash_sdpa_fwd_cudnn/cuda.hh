/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/flash_sdpa_fwd_cudnn/cuda.hh
 * Flash attention scaled dot-product attention forward pass using cuDNN
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <memory>
#include <cuda_runtime.h>
#include <cudnn_frontend.h>

namespace nntile::kernel::flash_sdpa_fwd_cudnn
{

// Shared pointer type for the graph
using FlashSdpaGraph = std::shared_ptr<cudnn_frontend::graph::Graph>;

//! Prepare cuDNN graph for flash attention
/*! Prepares and builds the cuDNN graph for flash attention with fixed dimensions.
 * The prepared graph can be reused for multiple executions with different data.
 * @param[in] handle: cuDNN handle (with stream already set)
 * @param[in] seq: Sequence length
 * @param[in] head: Head dimension (d_qk = d_v)
 * @param[in] batch: Batch size
 * @return Pointer to prepared graph structure, or nullptr on error
 * */
template<typename T>
FlashSdpaGraph prepare_graph(
    cudnnHandle_t handle,
    Index seq,
    Index head,
    Index batch
) noexcept;

//! Execute prepared cuDNN graph for flash attention
/*! Executes a previously prepared cuDNN graph with provided data pointers.
 * @param[in] handle: cuDNN handle (with stream already set)
 * @param[in] prepared_graph: Previously prepared graph structure
 * @param[in] K: Key tensor [batch, seq, head]
 * @param[in] Q: Query tensor [batch, seq, head]
 * @param[in] mask: Mask tensor [seq, seq]
 * @param[out] logsumexp: Log-sum-exp statistics [batch, seq]
 * @param[in] V: Value tensor [batch, seq, head]
 * @param[out] A: Attention output tensor [batch, seq, head]
 * */
template<typename T>
void execute_graph(
    cudnnHandle_t handle,
    const FlashSdpaGraph &prepared_graph,
    const T *K,
    const T *Q,
    const T *mask,
    fp32_t *logsumexp,
    const T *V,
    T *A
) noexcept;

} // namespace nntile::kernel::flash_sdpa_fwd_cudnn
