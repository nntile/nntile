/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tile/embedding_backward.cc
 * TileGraph embedding backward operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tile/embedding_backward.hh"
#include <stdexcept>
#include <nntile/base_types.hh>
#include <nntile/graph/dtype.hh>
#include <nntile/graph/tile.hh>
#include <nntile/tile/embedding_backward.hh>
namespace nntile::graph::tile_graph
{
namespace
{
template<typename T>
void run(
    TileGraph::Runtime& runtime, Index a, Index b, Index c, Index ks, Index kz, TileGraph::TileNode* i, TileGraph::TileNode* e, TileGraph::TileNode* v, int r)
{
    nntile::tile::embedding_backward<T>(a, b, c, ks, kz, runtime.get_tile<nntile::int64_t>(i), runtime.get_tile<T>(e), runtime.get_tile<T>(v), r);
}
} // namespace
void embedding_backward(
    Index m, Index n, Index k, Index k_start, Index k_size, TileGraph::TileNode* index, TileGraph::TileNode* embed, TileGraph::TileNode* vocab, int redux)
{
    if(!index || !embed || !vocab)
        throw std::invalid_argument("embedding_backward");
    if(index->graph() != embed->graph() || index->graph() != vocab->graph())
        throw std::invalid_argument("embedding_backward");
    if(index->dtype() != DataType::INT64)
        throw std::invalid_argument("embedding_backward");
    if(embed->dtype() != vocab->dtype())
        throw std::invalid_argument("embedding_backward");
    index->graph()->add_op(std::make_shared<TileEmbeddingBackwardOp>(
        m, n, k, k_start, k_size, index, embed, vocab, redux));
}
void TileEmbeddingBackwardOp::execute(TileGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(vocab);
    switch(dtype)
    {
        case DataType::FP32:
            run<nntile::fp32_t>(runtime, m, n, k, k_start, k_size, index, embed, vocab, redux);
            break;
        case DataType::FP32_FAST_TF32:
            run<nntile::fp32_fast_tf32_t>(runtime, m, n, k, k_start, k_size, index, embed, vocab, redux);
            break;
        case DataType::FP32_FAST_FP16:
            run<nntile::fp32_fast_fp16_t>(runtime, m, n, k, k_start, k_size, index, embed, vocab, redux);
            break;
        case DataType::FP32_FAST_BF16:
            run<nntile::fp32_fast_bf16_t>(runtime, m, n, k, k_start, k_size, index, embed, vocab, redux);
            break;
        case DataType::FP64:
            run<nntile::fp64_t>(runtime, m, n, k, k_start, k_size, index, embed, vocab, redux);
            break;
        case DataType::FP16:
            run<nntile::fp16_t>(runtime, m, n, k, k_start, k_size, index, embed, vocab, redux);
            break;
        case DataType::BF16:
            run<nntile::bf16_t>(runtime, m, n, k, k_start, k_size, index, embed, vocab, redux);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error("embedding_backward");
        default:
            throw std::runtime_error("embedding_backward");
    }
}
} // namespace nntile::graph::tile_graph
