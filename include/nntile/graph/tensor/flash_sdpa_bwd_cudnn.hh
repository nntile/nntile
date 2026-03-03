/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/flash_sdpa_bwd_cudnn.hh
 * TensorGraph flash_sdpa_bwd_cudnn: Flash SDPA backward (CUDA/cuDNN only)
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/graph/tensor/graph.hh>

namespace nntile::graph
{

//! Flash SDPA backward: dK, dQ, dV = backward(K, Q, V, A, dA, mask, logsumexp)
struct TensorFlashSdpaBwdCudnnOp : TensorGraph::OpNode
{
    TensorGraph::TensorNode* K = nullptr;
    TensorGraph::TensorNode* Q = nullptr;
    TensorGraph::TensorNode* V = nullptr;
    TensorGraph::TensorNode* A = nullptr;
    TensorGraph::TensorNode* dA = nullptr;
    TensorGraph::TensorNode* mask = nullptr;
    TensorGraph::TensorNode* logsumexp = nullptr;  // FP32
    TensorGraph::TensorNode* dK = nullptr;
    TensorGraph::TensorNode* dQ = nullptr;
    TensorGraph::TensorNode* dV = nullptr;

    TensorFlashSdpaBwdCudnnOp() = default;
    TensorFlashSdpaBwdCudnnOp(TensorGraph::TensorNode* K_,
                             TensorGraph::TensorNode* Q_,
                             TensorGraph::TensorNode* V_,
                             TensorGraph::TensorNode* A_,
                             TensorGraph::TensorNode* dA_,
                             TensorGraph::TensorNode* mask_,
                             TensorGraph::TensorNode* logsumexp_,
                             TensorGraph::TensorNode* dK_,
                             TensorGraph::TensorNode* dQ_,
                             TensorGraph::TensorNode* dV_)
        : K(K_), Q(Q_), V(V_), A(A_), dA(dA_), mask(mask_),
          logsumexp(logsumexp_), dK(dK_), dQ(dQ_), dV(dV_)
    {
        inputs_ = {
            K, Q, V, A, dA, mask, logsumexp,
            dK, dQ, dV
        };
        outputs_ = {dK, dQ, dV};
    }

    std::string op_name() const override { return "FLASH_SDPA_BWD_CUDNN"; }

    void execute(TensorGraph::Runtime& runtime) const override;

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorFlashSdpaBwdCudnnOp>(*this);
    }
};

//! Flash SDPA backward (CUDA only)
void flash_sdpa_bwd_cudnn(TensorGraph::TensorNode* K,
                          TensorGraph::TensorNode* Q,
                          TensorGraph::TensorNode* V,
                          TensorGraph::TensorNode* A,
                          TensorGraph::TensorNode* dA,
                          TensorGraph::TensorNode* mask,
                          TensorGraph::TensorNode* logsumexp,
                          TensorGraph::TensorNode* dK,
                          TensorGraph::TensorNode* dQ,
                          TensorGraph::TensorNode* dV);

} // namespace nntile::graph
