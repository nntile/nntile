/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tile/norm_fiber.cc
 * TileGraph norm fiber operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tile/norm_fiber.hh"
#include <stdexcept>
#include <nntile/base_types.hh>
#include <nntile/graph/dtype.hh>
#include <nntile/graph/tile.hh>
#include <nntile/tile/norm_fiber.hh>
namespace nntile::graph::tile_graph
{
namespace
{
template<typename T>
void run(
    TileGraph::Runtime& rt, Scalar a, TileGraph::TileNode* t1, Scalar b, TileGraph::TileNode* t2, TileGraph::TileNode* d, Index ax, Index bd, int r)
{
    nntile::tile::norm_fiber<T>(a, rt.get_tile<T>(t1), b, rt.get_tile<T>(t2), rt.get_tile<T>(d), ax, bd, r);
}
} // namespace
void norm_fiber(
    Scalar a, TileGraph::TileNode* t1, Scalar b, TileGraph::TileNode* t2, TileGraph::TileNode* d, Index ax, Index bd, int r)
{
    if(!t1 || !t2 || !d)
        throw std::invalid_argument("norm_fiber");
    if(t1->graph() != t2->graph() || t1->graph() != d->graph() || t1->dtype() != t2->dtype() || t1->dtype() != d->dtype())
        throw std::invalid_argument("norm_fiber");
    if(t1 == t2 || t1 == d || t2 == d)
        throw std::invalid_argument("norm_fiber");
    t1->graph()->add_op(std::make_shared<TileNormFiberOp>(a, t1, b, t2, d, ax, bd, r));
}
void TileNormFiberOp::execute(TileGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(s1);
    switch(dtype)
    {
        case DataType::FP32:
            run<nntile::fp32_t>(runtime, alpha, s1, beta, s2, dst, axis, batch_ndim, redux);
            break;
        case DataType::FP32_FAST_TF32:
            run<nntile::fp32_fast_tf32_t>(runtime, alpha, s1, beta, s2, dst, axis, batch_ndim, redux);
            break;
        case DataType::FP32_FAST_FP16:
            run<nntile::fp32_fast_fp16_t>(runtime, alpha, s1, beta, s2, dst, axis, batch_ndim, redux);
            break;
        case DataType::FP32_FAST_BF16:
            run<nntile::fp32_fast_bf16_t>(runtime, alpha, s1, beta, s2, dst, axis, batch_ndim, redux);
            break;
        case DataType::FP64:
            run<nntile::fp64_t>(runtime, alpha, s1, beta, s2, dst, axis, batch_ndim, redux);
            break;
        case DataType::FP16:
            throw std::runtime_error("FP16 not supported for tile norm_fiber in this build");
        case DataType::BF16:
            run<nntile::bf16_t>(runtime, alpha, s1, beta, s2, dst, axis, batch_ndim, redux);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error("norm_fiber");
        default:
            throw std::runtime_error("norm_fiber");
    }
}
} // namespace nntile::graph::tile_graph
