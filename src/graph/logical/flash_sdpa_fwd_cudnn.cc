/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/logical/flash_sdpa_fwd_cudnn.cc
 * Logical graph Flash SDPA forward CUDNN operation.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/logical/flash_sdpa_fwd_cudnn.hh"

// Include standard headers
#include <stdexcept>
#include <utility>

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! Flash attention forward pass (CUDA-only): A = flash_sdpa_fwd_cudnn(K, Q, mask, logsumexp, V)
void flash_sdpa_fwd_cudnn(
    LogicalGraph::TensorNode& K,
    LogicalGraph::TensorNode& Q,
    LogicalGraph::TensorNode& mask,
    LogicalGraph::TensorNode& logsumexp,
    LogicalGraph::TensorNode& V,
    LogicalGraph::TensorNode& A)
{
    if(&K.graph() != &Q.graph() || &K.graph() != &mask.graph() ||
       &K.graph() != &logsumexp.graph() || &K.graph() != &V.graph() || &K.graph() != &A.graph())
    {
        throw std::invalid_argument(
            "flash_sdpa_fwd_cudnn: tensors must belong to the same graph");
    }

    if(K.dtype() != Q.dtype() || K.dtype() != V.dtype() || K.dtype() != A.dtype())
    {
        throw std::invalid_argument(
            "flash_sdpa_fwd_cudnn: K, Q, V, A must have the same dtype");
    }

    if(logsumexp.dtype() != DataType::FP32)
    {
        throw std::invalid_argument(
            "flash_sdpa_fwd_cudnn: logsumexp must have fp32 dtype");
    }

    auto attrs = std::make_shared<ClearAttrs>(ClearAttrs{});
    K.graph().add_op(
        OpType::FLASH_SDPA_FWD_CUDNN,
        attrs,
        {&K, &Q, &mask, &logsumexp, &V},
        {&A}
    );
}

} // namespace nntile::graph
