/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tile/copy.cc
 * TileGraph copy operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tile/copy.hh"
#include <stdexcept>
#include <nntile/base_types.hh>
#include <nntile/graph/dtype.hh>
#include <nntile/graph/tile.hh>
#include <nntile/tile/copy.hh>
namespace nntile::graph::tile_graph
{
namespace
{
template<typename T>
void run_cp(TileGraph::Runtime& runtime, TileGraph::TileNode* s, TileGraph::TileNode* d)
{
    nntile::tile::copy<T>(runtime.get_tile<T>(s), runtime.get_tile<T>(d));
}
} // namespace
void copy(TileGraph::TileNode* src, TileGraph::TileNode* dst)
{
    if(!src || !dst)
        throw std::invalid_argument("tile copy: null");
    if(src->graph() != dst->graph())
        throw std::invalid_argument("tile copy: same graph");
    if(src->dtype() != dst->dtype())
        throw std::invalid_argument("tile copy: dtype");
    if(src == dst)
        throw std::invalid_argument("tile copy: distinct");
    src->graph()->add_op(std::make_shared<TileCopyOp>(src, dst));
}
void TileCopyOp::execute(TileGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(src);
    switch(dtype)
    {
        case DataType::FP32:
            run_cp<nntile::fp32_t>(runtime, src, dst);
            break;
        case DataType::FP32_FAST_TF32:
            run_cp<nntile::fp32_fast_tf32_t>(runtime, src, dst);
            break;
        case DataType::FP32_FAST_FP16:
            run_cp<nntile::fp32_fast_fp16_t>(runtime, src, dst);
            break;
        case DataType::FP32_FAST_BF16:
            run_cp<nntile::fp32_fast_bf16_t>(runtime, src, dst);
            break;
        case DataType::FP64:
            run_cp<nntile::fp64_t>(runtime, src, dst);
            break;
        case DataType::FP16:
            run_cp<nntile::fp16_t>(runtime, src, dst);
            break;
        case DataType::BF16:
            run_cp<nntile::bf16_t>(runtime, src, dst);
            break;
        case DataType::INT64:
            run_cp<nntile::int64_t>(runtime, src, dst);
            break;
        case DataType::BOOL:
            run_cp<nntile::bool_t>(runtime, src, dst);
            break;
        default:
            throw std::runtime_error("Unsupported data type for tile copy");
    }
}
} // namespace nntile::graph::tile_graph
