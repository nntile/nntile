/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/logical/flash_sdpa_fwd_cudnn.hh
 * Logical graph Flash SDPA forward CUDNN operation.
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

//! Flash attention forward pass (CUDA-only): A = flash_sdpa_fwd_cudnn(K, Q, mask, logsumexp, V)
//! @param K Key tensor
//! @param Q Query tensor
//! @param mask Attention mask tensor
//! @param logsumexp Log-sum-exp tensor (fp32)
//! @param V Value tensor
//! @param A Output attention tensor
void flash_sdpa_fwd_cudnn(
    LogicalGraph::TensorNode& K,
    LogicalGraph::TensorNode& Q,
    LogicalGraph::TensorNode& mask,
    LogicalGraph::TensorNode& logsumexp,
    LogicalGraph::TensorNode& V,
    LogicalGraph::TensorNode& A
);

} // namespace nntile::graph