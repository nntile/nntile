/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/logical/flash_sdpa_bwd_cudnn.hh
 * Logical graph Flash SDPA backward CUDNN operation.
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <string>

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! Flash attention backward pass (CUDA-only): gradients w.r.t. K, Q, V
//! @param K Key tensor
//! @param Q Query tensor
//! @param V Value tensor
//! @param A Forward pass attention output
//! @param dA Gradient of attention output
//! @param mask Attention mask tensor
//! @param logsumexp Log-sum-exp tensor (fp32)
//! @param dK Gradient tensor for K (modified in-place)
//! @param dQ Gradient tensor for Q (modified in-place)
//! @param dV Gradient tensor for V (modified in-place)
void flash_sdpa_bwd_cudnn(
    LogicalGraph::TensorNode& K,
    LogicalGraph::TensorNode& Q,
    LogicalGraph::TensorNode& V,
    LogicalGraph::TensorNode& A,
    LogicalGraph::TensorNode& dA,
    LogicalGraph::TensorNode& mask,
    LogicalGraph::TensorNode& logsumexp,
    LogicalGraph::TensorNode& dK,
    LogicalGraph::TensorNode& dQ,
    LogicalGraph::TensorNode& dV
);

} // namespace nntile::graph
