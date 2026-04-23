/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/flash_sdpa_fwd_cudnn.hh
 * TensorGraph flash_sdpa_fwd_cudnn: Flash SDPA forward (CUDA/cuDNN only)
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/graph/tensor/graph.hh>

namespace nntile::graph::tensor
{

//! Flash SDPA forward: A = attention(K, Q, V, mask)
struct TensorFlashSdpaFwdCudnnOp : TensorGraph::OpNode
{
    TensorGraph::TensorNode* K = nullptr;
    TensorGraph::TensorNode* Q = nullptr;
    TensorGraph::TensorNode* mask = nullptr;
    TensorGraph::TensorNode* logsumexp = nullptr;  // FP32
    TensorGraph::TensorNode* V = nullptr;
    TensorGraph::TensorNode* A = nullptr;

    TensorFlashSdpaFwdCudnnOp() = default;
    TensorFlashSdpaFwdCudnnOp(TensorGraph::TensorNode* K_,
                              TensorGraph::TensorNode* Q_,
                              TensorGraph::TensorNode* mask_,
                              TensorGraph::TensorNode* logsumexp_,
                              TensorGraph::TensorNode* V_,
                              TensorGraph::TensorNode* A_)
        : K(K_), Q(Q_), mask(mask_), logsumexp(logsumexp_), V(V_), A(A_)
    {
        inputs_ = {K, Q, mask, V};
        outputs_ = {logsumexp, A};
    }

    std::string op_name() const override { return "FLASH_SDPA_FWD_CUDNN"; }


    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorFlashSdpaFwdCudnnOp>(*this);
    }
};

//! Flash SDPA forward (CUDA only)
TensorGraph::TensorNode* flash_sdpa_fwd_cudnn(
    TensorGraph::TensorNode* K,
    TensorGraph::TensorNode* Q,
    TensorGraph::TensorNode* mask,
    TensorGraph::TensorNode* V,
    const std::string& logsumexp_name,
    const std::string& output_name);

void flash_sdpa_fwd_cudnn(TensorGraph::TensorNode* K,
                          TensorGraph::TensorNode* Q,
                          TensorGraph::TensorNode* mask,
                          TensorGraph::TensorNode* logsumexp,
                          TensorGraph::TensorNode* V,
                          TensorGraph::TensorNode* A);

} // namespace nntile::graph::tensor
