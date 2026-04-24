/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tile/maxsumexp.cc
 * TileGraph maxsumexp operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tile/maxsumexp.hh"
#include <stdexcept>
#include <nntile/base_types.hh>
#include <nntile/graph/dtype.hh>
#include <nntile/graph/tile.hh>
#include <nntile/tile/maxsumexp.hh>
namespace nntile::graph::tile_graph
{
namespace
{
template<typename T>
void run_me(
    TileGraph::Runtime& runtime, TileGraph::TileNode* s, TileGraph::TileNode* d, Index ax, int r)
{
    nntile::tile::maxsumexp<T>(runtime.get_tile<T>(s), runtime.get_tile<T>(d), ax, r);
}
} // namespace
void maxsumexp(TileGraph::TileNode* src, TileGraph::TileNode* dst, Index axis, int redux)
{
    if(!src || !dst)
        throw std::invalid_argument("tile maxsumexp: null");
    if(src->graph() != dst->graph())
        throw std::invalid_argument("tile maxsumexp: same graph");
    if(src->dtype() != dst->dtype())
        throw std::invalid_argument("tile maxsumexp: dtype");
    if(src == dst)
        throw std::invalid_argument("tile maxsumexp: distinct");
    src->graph()->add_op(std::make_shared<TileMaxsumexpOp>(src, dst, axis, redux));
}
void TileMaxsumexpOp::execute(TileGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(src);
    switch(dtype)
    {
        case DataType::FP32:
            run_me<nntile::fp32_t>(runtime, src, dst, axis, redux);
            break;
        case DataType::FP32_FAST_TF32:
            run_me<nntile::fp32_fast_tf32_t>(runtime, src, dst, axis, redux);
            break;
        case DataType::FP32_FAST_FP16:
            run_me<nntile::fp32_fast_fp16_t>(runtime, src, dst, axis, redux);
            break;
        case DataType::FP32_FAST_BF16:
            run_me<nntile::fp32_fast_bf16_t>(runtime, src, dst, axis, redux);
            break;
        case DataType::FP64:
            run_me<nntile::fp64_t>(runtime, src, dst, axis, redux);
            break;
        case DataType::FP16:
            run_me<nntile::fp16_t>(runtime, src, dst, axis, redux);
            break;
        case DataType::BF16:
            run_me<nntile::bf16_t>(runtime, src, dst, axis, redux);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) + " not supported for maxsumexp");
        default:
            throw std::runtime_error("Unsupported data type for maxsumexp");
    }
}
} // namespace nntile::graph::tile_graph
