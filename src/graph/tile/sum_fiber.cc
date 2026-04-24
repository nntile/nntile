/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tile/sum_fiber.cc
 * TileGraph sum fiber operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tile/sum_fiber.hh"
#include <stdexcept>
#include <nntile/base_types.hh>
#include <nntile/graph/dtype.hh>
#include <nntile/graph/tile.hh>
#include <nntile/tile/sum_fiber.hh>
namespace nntile::graph::tile_graph
{
namespace
{
template<typename T>
void run(
    TileGraph::Runtime& rt, Scalar a, TileGraph::TileNode* s, Scalar b, TileGraph::TileNode* d, Index ax, Index bnd, int r)
{
    nntile::tile::sum_fiber<T>(a, rt.get_tile<T>(s), b, rt.get_tile<T>(d), ax, bnd, r);
}
} // namespace
void sum_fiber(
    Scalar alpha, TileGraph::TileNode* src, Scalar beta, TileGraph::TileNode* dst, Index axis, Index batch_ndim, int redux)
{
    if(!src || !dst)
        throw std::invalid_argument("tile sum_fiber: null");
    if(src->graph() != dst->graph() || src->dtype() != dst->dtype() || src == dst)
        throw std::invalid_argument("tile sum_fiber: invalid");
    src->graph()->add_op(
        std::make_shared<TileSumFiberOp>(alpha, src, beta, dst, axis, batch_ndim, redux));
}
void TileSumFiberOp::execute(TileGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(src);
    switch(dtype)
    {
        case DataType::FP32:
            run<nntile::fp32_t>(runtime, alpha, src, beta, dst, axis, batch_ndim, redux);
            break;
        case DataType::FP32_FAST_TF32:
            run<nntile::fp32_fast_tf32_t>(runtime, alpha, src, beta, dst, axis, batch_ndim, redux);
            break;
        case DataType::FP32_FAST_FP16:
            run<nntile::fp32_fast_fp16_t>(runtime, alpha, src, beta, dst, axis, batch_ndim, redux);
            break;
        case DataType::FP32_FAST_BF16:
            run<nntile::fp32_fast_bf16_t>(runtime, alpha, src, beta, dst, axis, batch_ndim, redux);
            break;
        case DataType::FP64:
            run<nntile::fp64_t>(runtime, alpha, src, beta, dst, axis, batch_ndim, redux);
            break;
        case DataType::FP16:
            run<nntile::fp16_t>(runtime, alpha, src, beta, dst, axis, batch_ndim, redux);
            break;
        case DataType::BF16:
            run<nntile::bf16_t>(runtime, alpha, src, beta, dst, axis, batch_ndim, redux);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) + " not supported for sum_fiber");
        default:
            throw std::runtime_error("Unsupported data type for sum_fiber");
    }
}
} // namespace nntile::graph::tile_graph
