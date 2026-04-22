/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tile/gelu.cc
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tile/gelu.hh"

#include <stdexcept>

#include <nntile/base_types.hh>
#include <nntile/graph/dtype.hh>
#include <nntile/graph/tile.hh>
#include <nntile/tile/gelu.hh>

namespace nntile::graph::tile_graph
{

namespace
{

template<typename T>
void run_gelu(
    TileGraph::Runtime& runtime,
    TileGraph::TileNode* src,
    TileGraph::TileNode* dst)
{
    auto& s = runtime.get_tile<T>(src);
    auto& d = runtime.get_tile<T>(dst);
    nntile::tile::gelu<T>(s, d);
}

} // namespace

void gelu(TileGraph::TileNode* src, TileGraph::TileNode* dst)
{
    if(src == nullptr || dst == nullptr)
    {
        throw std::invalid_argument("tile gelu: src and dst must be non-null");
    }
    if(src->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "tile gelu: src and dst must belong to the same graph");
    }
    if(src->dtype() != dst->dtype())
    {
        throw std::invalid_argument("tile gelu: dtype mismatch");
    }
    if(src->shape() != dst->shape())
    {
        throw std::invalid_argument("tile gelu: shape mismatch");
    }

    auto op = std::make_shared<TileGeluOp>(src, dst);
    src->graph()->add_op(op);
}

void TileGeluOp::execute(TileGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(src);

    switch(dtype)
    {
        case DataType::FP32:
            run_gelu<nntile::fp32_t>(runtime, src, dst);
            break;
        case DataType::FP32_FAST_TF32:
            run_gelu<nntile::fp32_fast_tf32_t>(runtime, src, dst);
            break;
        case DataType::FP32_FAST_FP16:
            run_gelu<nntile::fp32_fast_fp16_t>(runtime, src, dst);
            break;
        case DataType::FP32_FAST_BF16:
            run_gelu<nntile::fp32_fast_bf16_t>(runtime, src, dst);
            break;
        case DataType::FP64:
            run_gelu<nntile::fp64_t>(runtime, src, dst);
            break;
        case DataType::FP16:
            run_gelu<nntile::fp16_t>(runtime, src, dst);
            break;
        case DataType::BF16:
            run_gelu<nntile::bf16_t>(runtime, src, dst);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for tile gelu");
        default:
            throw std::runtime_error("Unsupported data type for tile gelu");
    }
}

} // namespace nntile::graph::tile_graph
