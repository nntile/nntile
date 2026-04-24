/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tile/embedding.cc
 * TileGraph embedding operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tile/embedding.hh"
#include <stdexcept>
#include <nntile/base_types.hh>
#include <nntile/graph/dtype.hh>
#include <nntile/graph/tile.hh>
#include <nntile/tile/embedding.hh>
namespace nntile::graph::tile_graph
{
namespace
{
template<typename T>
void run(
    TileGraph::Runtime& runtime, Index a, Index b, Index c, Index ks, Index kz, TileGraph::TileNode* ix, TileGraph::TileNode* v, TileGraph::TileNode* e)
{
    nntile::tile::embedding<T>(a, b, c, ks, kz, runtime.get_tile<nntile::int64_t>(ix), runtime.get_tile<T>(v), runtime.get_tile<T>(e));
}
} // namespace
void embedding(
    Index m, Index n, Index k, Index k_start, Index k_size, TileGraph::TileNode* index, TileGraph::TileNode* vocab, TileGraph::TileNode* embed)
{
    if(!index || !vocab || !embed)
        throw std::invalid_argument("embedding");
    if(index->graph() != vocab->graph() || index->graph() != embed->graph())
        throw std::invalid_argument("embedding");
    if(index->dtype() != DataType::INT64)
        throw std::invalid_argument("embedding");
    if(vocab->dtype() != embed->dtype())
        throw std::invalid_argument("embedding");
    index->graph()->add_op(
        std::make_shared<TileEmbeddingOp>(m, n, k, k_start, k_size, index, vocab, embed));
}
void TileEmbeddingOp::execute(TileGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(vocab);
    switch(dtype)
    {
        case DataType::FP32:
            run<nntile::fp32_t>(runtime, m, n, k, k_start, k_size, index, vocab, embed);
            break;
        case DataType::FP32_FAST_TF32:
            run<nntile::fp32_fast_tf32_t>(runtime, m, n, k, k_start, k_size, index, vocab, embed);
            break;
        case DataType::FP32_FAST_FP16:
            run<nntile::fp32_fast_fp16_t>(runtime, m, n, k, k_start, k_size, index, vocab, embed);
            break;
        case DataType::FP32_FAST_BF16:
            run<nntile::fp32_fast_bf16_t>(runtime, m, n, k, k_start, k_size, index, vocab, embed);
            break;
        case DataType::FP64:
            run<nntile::fp64_t>(runtime, m, n, k, k_start, k_size, index, vocab, embed);
            break;
        case DataType::FP16:
            run<nntile::fp16_t>(runtime, m, n, k, k_start, k_size, index, vocab, embed);
            break;
        case DataType::BF16:
            run<nntile::bf16_t>(runtime, m, n, k, k_start, k_size, index, vocab, embed);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error("embedding");
        default:
            throw std::runtime_error("embedding");
    }
}
} // namespace nntile::graph::tile_graph
