/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tile/silu_inplace.cc
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tile/silu_inplace.hh"

#include <stdexcept>

#include <nntile/base_types.hh>
#include <nntile/graph/dtype.hh>
#include <nntile/graph/tile.hh>
#include <nntile/tile/silu_inplace.hh>

namespace nntile::graph::tile_graph
{

namespace
{

template<typename T>
void run_silu_inplace(TileGraph::Runtime& runtime, TileGraph::TileNode* d)
{
    auto& t = runtime.get_tile<T>(d);
    nntile::tile::silu_inplace<T>(t);
}

} // namespace

void silu_inplace(TileGraph::TileNode* dst)
{
    if(dst == nullptr)
    {
        throw std::invalid_argument("tile silu_inplace: dst must be non-null");
    }
    auto op = std::make_shared<TileSiluInplaceOp>(dst);
    dst->graph()->add_op(op);
}

void TileSiluInplaceOp::execute(TileGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(dst);

    switch(dtype)
    {
        case DataType::FP32:
            run_silu_inplace<nntile::fp32_t>(runtime, dst);
            break;
        case DataType::FP32_FAST_TF32:
            run_silu_inplace<nntile::fp32_fast_tf32_t>(runtime, dst);
            break;
        case DataType::FP32_FAST_FP16:
            run_silu_inplace<nntile::fp32_fast_fp16_t>(runtime, dst);
            break;
        case DataType::FP32_FAST_BF16:
            run_silu_inplace<nntile::fp32_fast_bf16_t>(runtime, dst);
            break;
        case DataType::FP64:
            run_silu_inplace<nntile::fp64_t>(runtime, dst);
            break;
        case DataType::FP16:
            run_silu_inplace<nntile::fp16_t>(runtime, dst);
            break;
        case DataType::BF16:
            run_silu_inplace<nntile::bf16_t>(runtime, dst);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for tile silu_inplace");
        default:
            throw std::runtime_error(
                "Unsupported data type for tile silu_inplace");
    }
}

} // namespace nntile::graph::tile_graph
