/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tile/sumprod_fiber.cc
 * TileGraph sumprod fiber operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tile/sumprod_fiber.hh"
#include <stdexcept>
#include <nntile/graph/dtype.hh>
#include <nntile/graph/tile.hh>
#include <nntile/tile/sumprod_fiber.hh>
namespace nntile::graph::tile_graph
{
namespace
{
template<typename T>
void run(
    TileGraph::Runtime& rt, Scalar a, TileGraph::TileNode* t1, TileGraph::TileNode* t2, Scalar b, TileGraph::TileNode* d, Index ax, int r)
{
    nntile::tile::sumprod_fiber<T>(a, rt.get_tile<T>(t1), rt.get_tile<T>(t2), b, rt.get_tile<T>(d), ax, r);
}
} // namespace
void sumprod_fiber(Scalar a, TileGraph::TileNode* t1, TileGraph::TileNode* t2, Scalar b, TileGraph::TileNode* dst, Index ax, int r)
{
    if(!t1 || !t2 || !dst)
        throw std::invalid_argument("sumprod_fiber");
    if(t1->graph() != t2->graph() || t1->graph() != dst->graph() || t1->dtype() != t2->dtype() || t1->dtype() != dst->dtype())
        throw std::invalid_argument("sumprod_fiber");
    if(t1 == t2 || t1 == dst || t2 == dst)
        throw std::invalid_argument("sumprod_fiber: distinct");
    t1->graph()->add_op(std::make_shared<TileSumprodFiberOp>(a, t1, t2, b, dst, ax, r));
}
void TileSumprodFiberOp::execute(TileGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(s1);
    switch(dtype)
    {
        case DataType::FP32:
            run<nntile::fp32_t>(runtime, alpha, s1, s2, beta, dst, axis, redux);
            break;
        case DataType::FP32_FAST_TF32:
            run<nntile::fp32_fast_tf32_t>(runtime, alpha, s1, s2, beta, dst, axis, redux);
            break;
        case DataType::FP32_FAST_FP16:
            run<nntile::fp32_fast_fp16_t>(runtime, alpha, s1, s2, beta, dst, axis, redux);
            break;
        case DataType::FP32_FAST_BF16:
            run<nntile::fp32_fast_bf16_t>(runtime, alpha, s1, s2, beta, dst, axis, redux);
            break;
        case DataType::FP64:
            run<nntile::fp64_t>(runtime, alpha, s1, s2, beta, dst, axis, redux);
            break;
        case DataType::FP16:
            run<nntile::fp16_t>(runtime, alpha, s1, s2, beta, dst, axis, redux);
            break;
        case DataType::BF16:
            run<nntile::bf16_t>(runtime, alpha, s1, s2, beta, dst, axis, redux);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(std::string(dtype_to_string(dtype)) + " for sumprod_fiber");
        default:
            throw std::runtime_error("sumprod_fiber");
    }
}
} // namespace nntile::graph::tile_graph
