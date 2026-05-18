/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/flash_sdpa_fwd_cudnn.hh
 * TileGraph flash_sdpa_fwd_cudnn: Flash SDPA forward (CUDA/cuDNN only)
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/graph/tile/graph.hh>

namespace nntile::graph::tile_graph
{

//! Flash SDPA forward: A = attention(K, Q, V, mask)
struct TileFlashSdpaFwdCudnnOp : TileGraph::OpNode
{
    TileGraph::TileNode *K = nullptr, *Q = nullptr, *mask = nullptr, *logsumexp = nullptr, *V = nullptr, *A = nullptr;
    TileFlashSdpaFwdCudnnOp() = default;
    TileFlashSdpaFwdCudnnOp(TileGraph::TileNode* a, TileGraph::TileNode* b, TileGraph::TileNode* m, TileGraph::TileNode* l, TileGraph::TileNode* v, TileGraph::TileNode* o) : K(a), Q(b), mask(m), logsumexp(l), V(v), A(o)
    {
        inputs_ = {K, Q, mask, logsumexp, V};
        outputs_ = {A};
    }
    std::string op_name() const override { return "TILE_FLASH_SDPA_FWD_CUDNN"; }
    void execute(TileGraph::Runtime& runtime) const override;
    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileFlashSdpaFwdCudnnOp>(*this);
    }
};
void flash_sdpa_fwd_cudnn(
    TileGraph::TileNode* K, TileGraph::TileNode* Q, TileGraph::TileNode* mask, TileGraph::TileNode* logsumexp, TileGraph::TileNode* V, TileGraph::TileNode* A);
} // namespace nntile::graph::tile_graph
