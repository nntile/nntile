/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tile/multiply_fiber.cc
 * TileGraph multiply fiber operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tile/multiply_fiber.hh"
#include <stdexcept>
#include <nntile/base_types.hh>
#include <nntile/graph/dtype.hh>
#include <nntile/graph/tile.hh>
#include <nntile/tile/multiply_fiber.hh>
namespace nntile::graph::tile_graph
{
namespace
{
template<typename T>
void run(
    TileGraph::Runtime& rt, Scalar a, TileGraph::TileNode* t1, TileGraph::TileNode* t2, TileGraph::TileNode* d, Index ax)
{
    nntile::tile::multiply_fiber<T>(a, rt.get_tile<T>(t1), rt.get_tile<T>(t2), rt.get_tile<T>(d), ax);
}
} // namespace
void multiply_fiber(Scalar a, TileGraph::TileNode* t1, TileGraph::TileNode* t2, TileGraph::TileNode* d, Index axis)
{
    if(!t1 || !t2 || !d)
        throw std::invalid_argument("multiply_fiber");
    if(t1->graph() != t2->graph() || t1->graph() != d->graph() || t1->dtype() != t2->dtype() || t1->dtype() != d->dtype())
        throw std::invalid_argument("multiply_fiber");
    if(t1 == t2 || t1 == d || t2 == d)
        throw std::invalid_argument("multiply_fiber");
    t1->graph()->add_op(std::make_shared<TileMultiplyFiberOp>(a, t1, t2, d, axis));
}
void TileMultiplyFiberOp::execute(TileGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(s1);
    switch(dtype)
    {
        case DataType::FP32:
            run<nntile::fp32_t>(runtime, alpha, s1, s2, dst, axis);
            break;
        case DataType::FP32_FAST_TF32:
            run<nntile::fp32_fast_tf32_t>(runtime, alpha, s1, s2, dst, axis);
            break;
        case DataType::FP32_FAST_FP16:
            run<nntile::fp32_fast_fp16_t>(runtime, alpha, s1, s2, dst, axis);
            break;
        case DataType::FP32_FAST_BF16:
            run<nntile::fp32_fast_bf16_t>(runtime, alpha, s1, s2, dst, axis);
            break;
        case DataType::FP64:
            run<nntile::fp64_t>(runtime, alpha, s1, s2, dst, axis);
            break;
        case DataType::FP16:
            run<nntile::fp16_t>(runtime, alpha, s1, s2, dst, axis);
            break;
        case DataType::BF16:
            run<nntile::bf16_t>(runtime, alpha, s1, s2, dst, axis);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error("multiply_fiber");
        default:
            throw std::runtime_error("multiply_fiber");
    }
}
} // namespace nntile::graph::tile_graph
