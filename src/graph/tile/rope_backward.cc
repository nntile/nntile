/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tile/rope_backward.cc
 * TileGraph rope backward operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tile/rope_backward.hh"
#include <stdexcept>
#include <nntile/base_types.hh>
#include <nntile/graph/dtype.hh>
#include <nntile/graph/tile.hh>
#include <nntile/tile/rope_backward.hh>
namespace nntile::graph::tile_graph
{
namespace
{
template<typename T>
void run(
    TileGraph::Runtime& runtime, TileGraph::TileNode* si, TileGraph::TileNode* co, TileGraph::TileNode* y, TileGraph::TileNode* x)
{
    nntile::tile::rope_backward<T>(runtime.get_tile<T>(si), runtime.get_tile<T>(co), runtime.get_tile<T>(y), runtime.get_tile<T>(x));
}
} // namespace
void rope_backward(TileGraph::TileNode* s1, TileGraph::TileNode* c, TileGraph::TileNode* y, TileGraph::TileNode* x)
{
    if(!s1 || !c || !y || !x)
        throw std::invalid_argument("rope_backward");
    if(s1->graph() != c->graph() || s1->graph() != y->graph() || s1->graph() != x->graph())
        throw std::invalid_argument("rope_backward");
    if(s1->dtype() != c->dtype() || s1->dtype() != y->dtype() || s1->dtype() != x->dtype())
        throw std::invalid_argument("rope_backward");
    if(s1 == c || s1 == y || s1 == x || c == y || c == x || y == x)
        throw std::invalid_argument("rope_backward");
    s1->graph()->add_op(std::make_shared<TileRopeBackwardOp>(s1, c, y, x));
}
void TileRopeBackwardOp::execute(TileGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(dy);
    switch(dtype)
    {
        case DataType::FP32:
            run<nntile::fp32_t>(runtime, sin, cos, dy, dx);
            break;
        case DataType::FP32_FAST_TF32:
            run<nntile::fp32_fast_tf32_t>(runtime, sin, cos, dy, dx);
            break;
        case DataType::FP32_FAST_FP16:
            run<nntile::fp32_fast_fp16_t>(runtime, sin, cos, dy, dx);
            break;
        case DataType::FP32_FAST_BF16:
            run<nntile::fp32_fast_bf16_t>(runtime, sin, cos, dy, dx);
            break;
        case DataType::FP64:
            run<nntile::fp64_t>(runtime, sin, cos, dy, dx);
            break;
        case DataType::FP16:
            run<nntile::fp16_t>(runtime, sin, cos, dy, dx);
            break;
        case DataType::BF16:
            run<nntile::bf16_t>(runtime, sin, cos, dy, dx);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error("rope_backward");
        default:
            throw std::runtime_error("rope_backward");
    }
}
} // namespace nntile::graph::tile_graph
