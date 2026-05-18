/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tile/sqrt.cc
 * TileGraph sqrt operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tile/sqrt.hh"

#include <stdexcept>

#include <nntile/base_types.hh>
#include <nntile/graph/dtype.hh>
#include <nntile/graph/tile.hh>
#include <nntile/tile/sqrt.hh>

namespace nntile::graph::tile_graph
{

namespace
{

template<typename T>
void run_sqrt(
    TileGraph::Runtime& runtime,
    TileGraph::TileNode* src,
    TileGraph::TileNode* dst)
{
    auto& s = runtime.get_tile<T>(src);
    auto& d = runtime.get_tile<T>(dst);
    nntile::tile::sqrt<T>(s, d);
}

} // namespace

void sqrt(TileGraph::TileNode* src, TileGraph::TileNode* dst)
{
    if(src == nullptr || dst == nullptr)
    {
        throw std::invalid_argument("tile sqrt: src and dst must be non-null");
    }
    if(src->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "tile sqrt: src and dst must belong to the same graph");
    }
    if(src->dtype() != dst->dtype())
    {
        throw std::invalid_argument("tile sqrt: dtype mismatch");
    }
    if(src->shape() != dst->shape())
    {
        throw std::invalid_argument("tile sqrt: shape mismatch");
    }

    auto op = std::make_shared<TileSqrtOp>(src, dst);
    src->graph()->add_op(op);
}

void TileSqrtOp::execute(TileGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(src);

    switch(dtype)
    {
        case DataType::FP32:
            run_sqrt<nntile::fp32_t>(runtime, src, dst);
            break;
        case DataType::FP32_FAST_TF32:
            run_sqrt<nntile::fp32_fast_tf32_t>(runtime, src, dst);
            break;
        case DataType::FP32_FAST_FP16:
            run_sqrt<nntile::fp32_fast_fp16_t>(runtime, src, dst);
            break;
        case DataType::FP32_FAST_BF16:
            run_sqrt<nntile::fp32_fast_bf16_t>(runtime, src, dst);
            break;
        case DataType::FP64:
            run_sqrt<nntile::fp64_t>(runtime, src, dst);
            break;
        case DataType::FP16:
            throw std::runtime_error(
                "FP16 data type not supported for tile sqrt");
        case DataType::BF16:
            run_sqrt<nntile::bf16_t>(runtime, src, dst);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for tile sqrt");
        default:
            throw std::runtime_error("Unsupported data type for tile sqrt");
    }
}

} // namespace nntile::graph::tile_graph
