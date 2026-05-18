/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tile/flash_sdpa_fwd_cudnn.cc
 * TileGraph flash sdpa fwd cudnn operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tile/flash_sdpa_fwd_cudnn.hh"
#include <stdexcept>
#include <nntile/base_types.hh>
#include <nntile/graph/dtype.hh>
#include <nntile/graph/tile.hh>
#include <nntile/tile/flash_sdpa_fwd_cudnn.hh>
namespace nntile::graph::tile_graph
{
namespace
{
template<typename T>
void run(
    TileGraph::Runtime& runtime, TileGraph::TileNode* K_, TileGraph::TileNode* Q_, TileGraph::TileNode* m, TileGraph::TileNode* l, TileGraph::TileNode* V_, TileGraph::TileNode* A_)
{
    nntile::tile::flash_sdpa_fwd_cudnn<T>(runtime.get_tile<T>(K_), runtime.get_tile<T>(Q_), runtime.get_tile<T>(m), runtime.get_tile<nntile::fp32_t>(l), runtime.get_tile<T>(V_), runtime.get_tile<T>(A_));
}
} // namespace
void flash_sdpa_fwd_cudnn(
    TileGraph::TileNode* K, TileGraph::TileNode* Q, TileGraph::TileNode* mask, TileGraph::TileNode* logsumexp, TileGraph::TileNode* V, TileGraph::TileNode* A)
{
    if(!K || !Q || !mask || !logsumexp || !V || !A)
        throw std::invalid_argument("flash_sdpa_fwd_cudnn");
    if(K->graph() != Q->graph() || K->graph() != mask->graph() || K->graph() != logsumexp->graph() || K->graph() != V->graph() || K->graph() != A->graph())
        throw std::invalid_argument("flash_sdpa_fwd_cudnn");
    if(K->dtype() != Q->dtype() || K->dtype() != mask->dtype() || K->dtype() != V->dtype() || K->dtype() != A->dtype())
        throw std::invalid_argument("flash_sdpa_fwd_cudnn");
    if(logsumexp->dtype() != DataType::FP32)
        throw std::invalid_argument("flash_sdpa_fwd_cudnn: logsumexp FP32");
    K->graph()->add_op(std::make_shared<TileFlashSdpaFwdCudnnOp>(K, Q, mask, logsumexp, V, A));
}
void TileFlashSdpaFwdCudnnOp::execute(TileGraph::Runtime& runtime) const
{
    // nntile::tile::flash_sdpa_fwd_cudnn is only explicitly instantiated for
    // fp16_t and bf16_t (see src/tile/flash_sdpa_fwd_cudnn.cc).
    DataType dtype = runtime.get_dtype(K);
    switch(dtype)
    {
        case DataType::FP16:
            run<nntile::fp16_t>(runtime, K, Q, mask, logsumexp, V, A);
            break;
        case DataType::BF16:
            run<nntile::bf16_t>(runtime, K, Q, mask, logsumexp, V, A);
            break;
        default:
            throw std::runtime_error("flash_sdpa_fwd_cudnn: only FP16 and BF16 are supported");
    }
}
} // namespace nntile::graph::tile_graph
