/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/logical/flash_sdpa_bwd_cudnn.cc
 * Logical graph Flash SDPA backward CUDNN operation.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/logical/flash_sdpa_bwd_cudnn.hh"

// Include standard headers
#include <stdexcept>
#include <utility>

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! Flash attention backward pass (CUDA-only): gradients w.r.t. K, Q, V = flash_sdpa_bwd_cudnn(...)
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
    LogicalGraph::TensorNode& dV)
{
    if(&K.graph() != &Q.graph() || &K.graph() != &V.graph() || &K.graph() != &A.graph() ||
       &K.graph() != &dA.graph() || &K.graph() != &mask.graph() || &K.graph() != &logsumexp.graph() ||
       &K.graph() != &dK.graph() || &K.graph() != &dQ.graph() || &K.graph() != &dV.graph())
    {
        throw std::invalid_argument(
            "flash_sdpa_bwd_cudnn: tensors must belong to the same graph");
    }

    if(K.dtype() != Q.dtype() || K.dtype() != V.dtype() || K.dtype() != A.dtype() ||
       K.dtype() != dA.dtype() || K.dtype() != dK.dtype() || K.dtype() != dQ.dtype() || K.dtype() != dV.dtype())
    {
        throw std::invalid_argument(
            "flash_sdpa_bwd_cudnn: all tensors must have the same dtype");
    }

    if(logsumexp.dtype() != DataType::FP32)
    {
        throw std::invalid_argument(
            "flash_sdpa_bwd_cudnn: logsumexp must have fp32 dtype");
    }

    OpAttrs attrs = ClearAttrs{};  // No additional attributes needed
    K.graph().add_op(
        OpType::FLASH_SDPA_BWD_CUDNN,
        attrs,
        {&K, &Q, &V, &A, &dA, &mask, &logsumexp, &dK, &dQ, &dV},
        {&dK, &dQ, &dV}
    );
}

} // namespace nntile::graph
