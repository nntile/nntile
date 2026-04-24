/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tile/add_fiber.cc
 * TileGraph add fiber operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tile/add_fiber.hh"
#include <stdexcept>
#include <nntile/base_types.hh>
#include <nntile/graph/dtype.hh>
#include <nntile/graph/tile.hh>
#include <nntile/tile/add_fiber.hh>
namespace nntile::graph::tile_graph
{
namespace
{
template<typename T>
void run(
    TileGraph::Runtime& runtime,
    Scalar a,
    TileGraph::TileNode* t1,
    Scalar b,
    TileGraph::TileNode* t2,
    TileGraph::TileNode* d,
    Index ax,
    Index bd)
{
    nntile::tile::add_fiber<T>(
        a, runtime.get_tile<T>(t1), b, runtime.get_tile<T>(t2), runtime.get_tile<T>(d), ax, bd);
}
} // namespace
void add_fiber(
    Scalar a, TileGraph::TileNode* s1, Scalar b, TileGraph::TileNode* s2, TileGraph::TileNode* dst, Index axis, Index batch_ndim)
{
    if(!s1 || !s2 || !dst)
        throw std::invalid_argument("add_fiber");
    if(s1->graph() != s2->graph() || s1->graph() != dst->graph() || s1->dtype() != s2->dtype() || s1->dtype() != dst->dtype())
        throw std::invalid_argument("add_fiber");
    if(s1 == s2 || s1 == dst || s2 == dst)
        throw std::invalid_argument("add_fiber");
    s1->graph()->add_op(
        std::make_shared<TileAddFiberOp>(a, s1, b, s2, dst, axis, batch_ndim));
}
void TileAddFiberOp::execute(TileGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(s1);
    switch(dtype)
    {
        case DataType::FP32:
            run<nntile::fp32_t>(runtime, alpha, s1, beta, s2, dst, axis, batch_ndim);
            break;
        case DataType::FP32_FAST_TF32:
            run<nntile::fp32_fast_tf32_t>(runtime, alpha, s1, beta, s2, dst, axis, batch_ndim);
            break;
        case DataType::FP32_FAST_FP16:
            run<nntile::fp32_fast_fp16_t>(runtime, alpha, s1, beta, s2, dst, axis, batch_ndim);
            break;
        case DataType::FP32_FAST_BF16:
            run<nntile::fp32_fast_bf16_t>(runtime, alpha, s1, beta, s2, dst, axis, batch_ndim);
            break;
        case DataType::FP64:
            run<nntile::fp64_t>(runtime, alpha, s1, beta, s2, dst, axis, batch_ndim);
            break;
        case DataType::FP16:
            throw std::runtime_error("FP16 not supported for add_fiber in this build");
        case DataType::BF16:
            run<nntile::bf16_t>(runtime, alpha, s1, beta, s2, dst, axis, batch_ndim);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error("dtype for add_fiber");
        default:
            throw std::runtime_error("add_fiber");
    }
}
} // namespace nntile::graph::tile_graph
