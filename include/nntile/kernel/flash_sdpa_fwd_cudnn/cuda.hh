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
#include <cuda_runtime.h>
#include <cudnn.h>

namespace nntile::kernel::flash_sdpa_fwd_cudnn
{

//! Forward declaration of prepared graph structure
template<typename T>
struct FlashSdpaGraph;

//! Prepare cuDNN graph for flash attention
/*! Prepares and builds the cuDNN graph for flash attention with fixed dimensions.
 * The prepared graph can be reused for multiple executions with different data.
 * @param[in] handle: cuDNN handle (with stream already set)
 * @param[in] seq: Sequence length
 * @param[in] head: Head dimension (d_qk = d_v)
 * @param[in] batch: Batch size
 * @param[in] use_mask: Whether to use custom mask (true) or causal mask (false)
 * @return Pointer to prepared graph structure, or nullptr on error
 * */
template<typename T>
FlashSdpaGraph<T>* prepare_graph(
    cudnnHandle_t handle,
    Index seq,
    Index head,
    Index batch,
    bool use_mask
) noexcept;

//! Execute prepared cuDNN graph for flash attention
/*! Executes a previously prepared cuDNN graph with provided data pointers.
 * @param[in] handle: cuDNN handle (with stream already set)
 * @param[in] prepared_graph: Previously prepared graph structure
 * @param[in] K: Key tensor [batch, seq, head]
 * @param[in] Q: Query tensor [batch, seq, head]
 * @param[in] mask: Optional mask tensor [batch, seq, seq]. Required if graph was prepared with use_mask=true
 * @param[out] logsumexp: Log-sum-exp statistics [batch, seq]
 * @param[in] V: Value tensor [batch, seq, head]
 * @param[out] A: Attention output tensor [batch, seq, head]
 * */
template<typename T>
void execute_graph(
    cudnnHandle_t handle,
    const FlashSdpaGraph<T> *prepared_graph,
    const T *K,
    const T *Q,
    const T *mask,
    T *logsumexp,
    const T *V,
    T *A
) noexcept;

//! Destroy prepared cuDNN graph
/*! Frees resources associated with a prepared graph.
 * @param[in] prepared_graph: Graph structure to destroy
 * */
template<typename T>
void destroy_graph(
    FlashSdpaGraph<T> *prepared_graph
) noexcept;

//! Flash attention forward pass using cuDNN (convenience wrapper)
/*! Performs scaled dot-product attention using cuDNN's Flash Attention implementation.
 * This is a convenience function that prepares the graph, executes it, and destroys it.
 * For better performance with repeated calls, use prepare_graph/execute_graph/destroy_graph directly.
 * @param[in] handle: cuDNN handle (with stream already set)
 * @param[in] seq: Sequence length
 * @param[in] head: Head dimension (d_qk = d_v)
 * @param[in] batch: Batch size
 * @param[in] K: Key tensor [batch, seq, head]
 * @param[in] Q: Query tensor [batch, seq, head]
 * @param[in] mask: Optional mask tensor [batch, seq, seq] (boolean). If nullptr, uses causal mask
 * @param[out] logsumexp: Log-sum-exp statistics [batch, seq]
 * @param[in] V: Value tensor [batch, seq, head]
 * @param[out] A: Attention output tensor [batch, seq, head]
 * */
template<typename T>
void cuda(
    cudnnHandle_t handle,
    Index seq,
    Index head,
    Index batch,
    const T *K,
    const T *Q,
    const T *mask,
    T *logsumexp,
    const T *V,
    T *A
) noexcept;

} // namespace nntile::kernel::flash_sdpa_fwd_cudnn
