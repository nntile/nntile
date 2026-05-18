/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tile/subtract_indexed_outputs.cc
 * TileGraph subtract indexed outputs operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tile/subtract_indexed_outputs.hh"
#include <stdexcept>
#include <nntile/base_types.hh>
#include <nntile/graph/dtype.hh>
#include <nntile/graph/tile.hh>
#include <nntile/tile/subtract_indexed_outputs.hh>
namespace nntile::graph::tile_graph
{
namespace
{
template<typename T>
void run(
    TileGraph::Runtime& runtime, Scalar vl, TileGraph::TileNode* l, TileGraph::TileNode* d, Index ig)
{
    nntile::tile::subtract_indexed_outputs<T>(vl, runtime.get_tile<nntile::int64_t>(l), runtime.get_tile<T>(d), ig);
}
} // namespace
void subtract_indexed_outputs(Scalar v, TileGraph::TileNode* labels, TileGraph::TileNode* dst, Index ignore_index)
{
    if(!labels || !dst)
        throw std::invalid_argument("subtract_indexed_outputs");
    if(labels->graph() != dst->graph() || labels->dtype() != DataType::INT64)
        throw std::invalid_argument("subtract_indexed_outputs");
    dst->graph()->add_op(
        std::make_shared<TileSubtractIndexedOutputsOp>(v, labels, dst, ignore_index));
}
void TileSubtractIndexedOutputsOp::execute(TileGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(dst);
    switch(dtype)
    {
        case DataType::FP32:
            run<nntile::fp32_t>(runtime, v, labels, dst, ignore_index);
            break;
        case DataType::FP32_FAST_TF32:
            run<nntile::fp32_fast_tf32_t>(runtime, v, labels, dst, ignore_index);
            break;
        case DataType::FP32_FAST_FP16:
            run<nntile::fp32_fast_fp16_t>(runtime, v, labels, dst, ignore_index);
            break;
        case DataType::FP32_FAST_BF16:
            run<nntile::fp32_fast_bf16_t>(runtime, v, labels, dst, ignore_index);
            break;
        case DataType::FP64:
            run<nntile::fp64_t>(runtime, v, labels, dst, ignore_index);
            break;
        case DataType::FP16:
            run<nntile::fp16_t>(runtime, v, labels, dst, ignore_index);
            break;
        case DataType::BF16:
            run<nntile::bf16_t>(runtime, v, labels, dst, ignore_index);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error("subtract_indexed_outputs");
        default:
            throw std::runtime_error("subtract_indexed_outputs");
    }
}
} // namespace nntile::graph::tile_graph
