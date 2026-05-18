/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/flash_sdpa_bwd_cudnn.hh
 * TileGraph flash_sdpa_bwd_cudnn: Flash SDPA backward (CUDA/cuDNN only)
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/graph/tile/graph.hh>

namespace nntile::graph::tile_graph
{

//! Flash SDPA backward: dK, dQ, dV = backward(K, Q, V, A, dA, mask, logsumexp)
struct TileFlashSdpaBwdCudnnOp : TileGraph::OpNode
{
    TileGraph::TileNode* K = nullptr;
    TileGraph::TileNode* Q = nullptr;
    TileGraph::TileNode* V = nullptr;
    TileGraph::TileNode* A = nullptr;
    TileGraph::TileNode* dA_ = nullptr;
    TileGraph::TileNode* mask = nullptr;
    TileGraph::TileNode* logsumexp = nullptr;
    TileGraph::TileNode* dK = nullptr;
    TileGraph::TileNode* dQ = nullptr;
    TileGraph::TileNode* dV = nullptr;
    TileFlashSdpaBwdCudnnOp() = default;
    TileFlashSdpaBwdCudnnOp(
        TileGraph::TileNode* a,
        TileGraph::TileNode* b,
        TileGraph::TileNode* c,
        TileGraph::TileNode* d,
        TileGraph::TileNode* e,
        TileGraph::TileNode* f,
        TileGraph::TileNode* g,
        TileGraph::TileNode* h,
        TileGraph::TileNode* i,
        TileGraph::TileNode* j) : K(a), Q(b), V(c), A(d), dA_(e), mask(f), logsumexp(g), dK(h), dQ(i), dV(j)
    {
        inputs_ = {K, Q, V, A, dA_, mask, logsumexp, dK, dQ, dV};
        outputs_ = {dK, dQ, dV};
    }
    std::string op_name() const override { return "TILE_FLASH_SDPA_BWD_CUDNN"; }
    void execute(TileGraph::Runtime& runtime) const override;
    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileFlashSdpaBwdCudnnOp>(*this);
    }
};
//! Flash SDPA backward (CUDA only)
void flash_sdpa_bwd_cudnn(
    TileGraph::TileNode* K,
    TileGraph::TileNode* Q,
    TileGraph::TileNode* V,
    TileGraph::TileNode* A,
    TileGraph::TileNode* dA,
    TileGraph::TileNode* mask,
    TileGraph::TileNode* logsumexp,
    TileGraph::TileNode* dK,
    TileGraph::TileNode* dQ,
    TileGraph::TileNode* dV);
} // namespace nntile::graph::tile_graph
