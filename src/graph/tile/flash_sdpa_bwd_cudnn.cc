/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tile/flash_sdpa_bwd_cudnn.cc
 * TileGraph flash sdpa bwd cudnn operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tile/flash_sdpa_bwd_cudnn.hh"
#include <stdexcept>
#include <nntile/base_types.hh>
#include <nntile/graph/dtype.hh>
#include <nntile/graph/tile.hh>
#include <nntile/tile/flash_sdpa_bwd_cudnn.hh>
namespace nntile::graph::tile_graph
{
namespace
{
template<typename T>
void run(
    TileGraph::Runtime& runtime,
    TileGraph::TileNode* K_,
    TileGraph::TileNode* Q_,
    TileGraph::TileNode* V_,
    TileGraph::TileNode* A_,
    TileGraph::TileNode* dA,
    TileGraph::TileNode* m,
    TileGraph::TileNode* lse,
    TileGraph::TileNode* dK_,
    TileGraph::TileNode* dQ_,
    TileGraph::TileNode* dV_)
{
    nntile::tile::flash_sdpa_bwd_cudnn<T>(runtime.get_tile<T>(K_), runtime.get_tile<T>(Q_), runtime.get_tile<T>(V_), runtime.get_tile<T>(A_), runtime.get_tile<T>(dA), runtime.get_tile<T>(m), runtime.get_tile<nntile::fp32_t>(lse), runtime.get_tile<T>(dK_), runtime.get_tile<T>(dQ_), runtime.get_tile<T>(dV_));
}
} // namespace
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
    TileGraph::TileNode* dV)
{
    if(!K || !Q || !V || !A || !dA || !mask || !logsumexp || !dK || !dQ || !dV)
        throw std::invalid_argument("flash_sdpa_bwd_cudnn");
    if(K->graph() != Q->graph() || K->graph() != V->graph() || K->graph() != A->graph() || K->graph() != dA->graph() || K->graph() != mask->graph() || K->graph() != logsumexp->graph() || K->graph() != dK->graph() || K->graph() != dQ->graph() || K->graph() != dV->graph())
        throw std::invalid_argument("flash_sdpa_bwd_cudnn");
    if(K->dtype() != Q->dtype() || K->dtype() != V->dtype() || K->dtype() != A->dtype() || K->dtype() != dA->dtype() || K->dtype() != mask->dtype() || K->dtype() != dK->dtype() || K->dtype() != dQ->dtype() || K->dtype() != dV->dtype())
        throw std::invalid_argument("flash_sdpa_bwd_cudnn");
    if(logsumexp->dtype() != DataType::FP32)
        throw std::invalid_argument("flash_sdpa_bwd_cudnn");
    K->graph()->add_op(
        std::make_shared<TileFlashSdpaBwdCudnnOp>(K, Q, V, A, dA, mask, logsumexp, dK, dQ, dV));
}
void TileFlashSdpaBwdCudnnOp::execute(TileGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(K);
    switch(dtype)
    {
        case DataType::FP16:
            run<nntile::fp16_t>(runtime, K, Q, V, A, dA_, mask, logsumexp, dK, dQ, dV);
            break;
        case DataType::BF16:
            run<nntile::bf16_t>(runtime, K, Q, V, A, dA_, mask, logsumexp, dK, dQ, dV);
            break;
        default:
            throw std::runtime_error("flash_sdpa_bwd_cudnn: only FP16 and BF16 are supported");
    }
}
} // namespace nntile::graph::tile_graph
