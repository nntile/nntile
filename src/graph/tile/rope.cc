/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tile/rope.cc
 * TileGraph rope operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tile/rope.hh"
#include <stdexcept>
#include <nntile/base_types.hh>
#include <nntile/graph/dtype.hh>
#include <nntile/graph/tile.hh>
#include <nntile/tile/rope.hh>
namespace nntile::graph::tile_graph
{
namespace
{
template<typename T>
void run(
    TileGraph::Runtime& runtime, TileGraph::TileNode* si, TileGraph::TileNode* co, TileGraph::TileNode* s, TileGraph::TileNode* d)
{
    nntile::tile::rope<T>(runtime.get_tile<T>(si), runtime.get_tile<T>(co), runtime.get_tile<T>(s), runtime.get_tile<T>(d));
}
} // namespace
void rope(TileGraph::TileNode* s1, TileGraph::TileNode* c, TileGraph::TileNode* s, TileGraph::TileNode* d)
{
    if(!s1 || !c || !s || !d)
        throw std::invalid_argument("rope");
    if(s1->graph() != c->graph() || s1->graph() != s->graph() || s1->graph() != d->graph())
        throw std::invalid_argument("rope");
    if(s1->dtype() != c->dtype() || s1->dtype() != s->dtype() || s1->dtype() != d->dtype())
        throw std::invalid_argument("rope");
    if(s1 == c || s1 == s || s1 == d || c == s || c == d || s == d)
        throw std::invalid_argument("rope");
    s1->graph()->add_op(std::make_shared<TileRopeOp>(s1, c, s, d));
}
void TileRopeOp::execute(TileGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(src);
    switch(dtype)
    {
        case DataType::FP32:
            run<nntile::fp32_t>(runtime, sin, cos, src, dst);
            break;
        case DataType::FP32_FAST_TF32:
            run<nntile::fp32_fast_tf32_t>(runtime, sin, cos, src, dst);
            break;
        case DataType::FP32_FAST_FP16:
            run<nntile::fp32_fast_fp16_t>(runtime, sin, cos, src, dst);
            break;
        case DataType::FP32_FAST_BF16:
            run<nntile::fp32_fast_bf16_t>(runtime, sin, cos, src, dst);
            break;
        case DataType::FP64:
            run<nntile::fp64_t>(runtime, sin, cos, src, dst);
            break;
        case DataType::FP16:
            run<nntile::fp16_t>(runtime, sin, cos, src, dst);
            break;
        case DataType::BF16:
            run<nntile::bf16_t>(runtime, sin, cos, src, dst);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error("rope");
        default:
            throw std::runtime_error("rope");
    }
}
} // namespace nntile::graph::tile_graph
