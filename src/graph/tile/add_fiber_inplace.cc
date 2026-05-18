/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tile/add_fiber_inplace.cc
 * TileGraph add fiber inplace operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tile/add_fiber_inplace.hh"
#include <stdexcept>
#include <nntile/base_types.hh>
#include <nntile/graph/dtype.hh>
#include <nntile/graph/tile.hh>
#include <nntile/tile/add_fiber_inplace.hh>
namespace nntile::graph::tile_graph
{
namespace
{
template<typename T>
void run(
    TileGraph::Runtime& rt, Scalar a, TileGraph::TileNode* s, Scalar b, TileGraph::TileNode* d, Index ax, Index bd)
{
    nntile::tile::add_fiber_inplace<T>(a, rt.get_tile<T>(s), b, rt.get_tile<T>(d), ax, bd);
}
} // namespace
void add_fiber_inplace(Scalar a, TileGraph::TileNode* s, Scalar b, TileGraph::TileNode* d, Index axis, Index batch_ndim)
{
    if(!s || !d)
        throw std::invalid_argument("add_fiber_inplace");
    if(s->graph() != d->graph() || s->dtype() != d->dtype() || s == d)
        throw std::invalid_argument("add_fiber_inplace");
    s->graph()->add_op(std::make_shared<TileAddFiberInplaceOp>(a, s, b, d, axis, batch_ndim));
}
void TileAddFiberInplaceOp::execute(TileGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(src);
    switch(dtype)
    {
        case DataType::FP32:
            run<nntile::fp32_t>(runtime, alpha, src, beta, dst, axis, batch_ndim);
            break;
        case DataType::FP32_FAST_TF32:
            run<nntile::fp32_fast_tf32_t>(runtime, alpha, src, beta, dst, axis, batch_ndim);
            break;
        case DataType::FP32_FAST_FP16:
            run<nntile::fp32_fast_fp16_t>(runtime, alpha, src, beta, dst, axis, batch_ndim);
            break;
        case DataType::FP32_FAST_BF16:
            run<nntile::fp32_fast_bf16_t>(runtime, alpha, src, beta, dst, axis, batch_ndim);
            break;
        case DataType::FP64:
            run<nntile::fp64_t>(runtime, alpha, src, beta, dst, axis, batch_ndim);
            break;
        case DataType::FP16:
            run<nntile::fp16_t>(runtime, alpha, src, beta, dst, axis, batch_ndim);
            break;
        case DataType::BF16:
            run<nntile::bf16_t>(runtime, alpha, src, beta, dst, axis, batch_ndim);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error("add_fiber_inplace");
        default:
            throw std::runtime_error("add_fiber_inplace");
    }
}
} // namespace nntile::graph::tile_graph
