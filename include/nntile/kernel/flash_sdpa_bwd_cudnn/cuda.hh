/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/flash_sdpa_bwd_cudnn/cuda.hh
 * Flash attention scaled dot-product attention backward pass using cuDNN
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <memory>
#include <cuda_runtime.h>
#include <cudnn_frontend.h>

namespace nntile::kernel::flash_sdpa_bwd_cudnn
{

// Shared pointer type for the graph
using FlashSdpaGraph = std::shared_ptr<cudnn_frontend::graph::Graph>;

//! Prepare cuDNN graph for flash attention backward pass
template<typename T>
FlashSdpaGraph prepare_graph(
    cudnnHandle_t handle,
    Index seq,
    Index head,
    Index batch
) noexcept;

//! Execute prepared cuDNN graph for flash attention backward pass
template<typename T>
void execute_graph(
    cudnnHandle_t handle,
    const FlashSdpaGraph &prepared_graph,
    Index seq,
    Index head,
    Index batch,
    const T *K,
    const T *Q,
    const T *V,
    const T *O,
    const T *dO,
    const T *mask,
    const fp32_t *logsumexp,
    T *scratch_dK,
    T *scratch_dQ,
    T *scratch_dV,
    T *dK,
    T *dQ,
    T *dV,
    void *workspace
) noexcept;

} // namespace nntile::kernel::flash_sdpa_bwd_cudnn
