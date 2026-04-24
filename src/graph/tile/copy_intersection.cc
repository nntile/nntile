/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tile/copy_intersection.cc
 * TileGraph copy intersection operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tile/copy_intersection.hh"

#include <stdexcept>
#include <vector>

#include <nntile/base_types.hh>
#include <nntile/graph/dtype.hh>
#include <nntile/graph/tile.hh>
#include <nntile/tile/copy_intersection.hh>

namespace nntile::graph::tile_graph
{
namespace
{
template<typename T>
void run(
    TileGraph::Runtime& runtime,
    const std::vector<Index>& so,
    const std::vector<Index>& df,
    TileGraph::TileNode* s,
    TileGraph::TileNode* d,
    TileGraph::TileNode* sc)
{
    nntile::tile::copy_intersection<T>(
        runtime.get_tile<T>(s), so, runtime.get_tile<T>(d), df, runtime.get_tile<nntile::int64_t>(sc));
}
} // namespace
void copy_intersection(
    TileGraph::TileNode* src, const std::vector<Index>& src_offset, TileGraph::TileNode* dst, const std::vector<Index>& dst_offset, TileGraph::TileNode* scratch)
{
    if(!src || !dst || !scratch)
        throw std::invalid_argument("copy_intersection");
    if(src->graph() != dst->graph() || src->graph() != scratch->graph())
        throw std::invalid_argument("copy_intersection");
    if(src->dtype() != dst->dtype() || scratch->dtype() != DataType::INT64)
        throw std::invalid_argument("copy_intersection");
    if(src == dst)
        throw std::invalid_argument("copy_intersection");
    src->graph()->add_op(
        std::make_shared<TileCopyIntersectionOp>(src_offset, dst_offset, src, dst, scratch));
}
void TileCopyIntersectionOp::execute(TileGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(src);
    switch(dtype)
    {
        case DataType::FP32:
            run<nntile::fp32_t>(runtime, src_offset, dst_offset, src, dst, scratch);
            break;
        case DataType::FP32_FAST_TF32:
            run<nntile::fp32_fast_tf32_t>(runtime, src_offset, dst_offset, src, dst, scratch);
            break;
        case DataType::FP32_FAST_FP16:
            run<nntile::fp32_fast_fp16_t>(runtime, src_offset, dst_offset, src, dst, scratch);
            break;
        case DataType::FP32_FAST_BF16:
            run<nntile::fp32_fast_bf16_t>(runtime, src_offset, dst_offset, src, dst, scratch);
            break;
        case DataType::FP64:
            run<nntile::fp64_t>(runtime, src_offset, dst_offset, src, dst, scratch);
            break;
        case DataType::FP16:
            run<nntile::fp16_t>(runtime, src_offset, dst_offset, src, dst, scratch);
            break;
        case DataType::BF16:
            run<nntile::bf16_t>(runtime, src_offset, dst_offset, src, dst, scratch);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error("copy_intersection");
        default:
            throw std::runtime_error("copy_intersection");
    }
}
} // namespace nntile::graph::tile_graph
