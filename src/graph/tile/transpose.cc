/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tile/transpose.cc
 * TileGraph transpose operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tile/transpose.hh"

#include <stdexcept>

#include <nntile/base_types.hh>
#include <nntile/graph/dtype.hh>
#include <nntile/graph/tile.hh>
#include <nntile/tile/transpose.hh>

namespace nntile::graph::tile_graph
{

namespace
{

template<typename T>
void run_transpose(
    TileGraph::Runtime& runtime, Scalar alpha, Index ndim, TileGraph::TileNode* src, TileGraph::TileNode* dst)
{
    auto& s = runtime.get_tile<T>(src);
    auto& d = runtime.get_tile<T>(dst);
    nntile::tile::transpose<T>(alpha, s, d, ndim);
}

} // namespace

void transpose(Scalar alpha, TileGraph::TileNode* src, TileGraph::TileNode* dst, Index ndim)
{
    if(src == nullptr || dst == nullptr)
    {
        throw std::invalid_argument("tile transpose: src and dst must be non-null");
    }
    if(src->graph() != dst->graph())
    {
        throw std::invalid_argument("tile transpose: src and dst must belong to the same graph");
    }
    if(src->dtype() != dst->dtype())
    {
        throw std::invalid_argument("tile transpose: dtype mismatch");
    }
    if(src == dst)
    {
        throw std::invalid_argument("tile transpose: src and dst must be distinct");
    }
    auto op = std::make_shared<TileTransposeOp>(alpha, src, dst, ndim);
    src->graph()->add_op(op);
}

void TileTransposeOp::execute(TileGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(src);
    switch(dtype)
    {
        case DataType::FP32:
            run_transpose<nntile::fp32_t>(runtime, alpha, ndim, src, dst);
            break;
        case DataType::FP32_FAST_TF32:
            run_transpose<nntile::fp32_fast_tf32_t>(runtime, alpha, ndim, src, dst);
            break;
        case DataType::FP32_FAST_FP16:
            run_transpose<nntile::fp32_fast_fp16_t>(runtime, alpha, ndim, src, dst);
            break;
        case DataType::FP32_FAST_BF16:
            run_transpose<nntile::fp32_fast_bf16_t>(runtime, alpha, ndim, src, dst);
            break;
        case DataType::FP64:
            run_transpose<nntile::fp64_t>(runtime, alpha, ndim, src, dst);
            break;
        case DataType::FP16:
            run_transpose<nntile::fp16_t>(runtime, alpha, ndim, src, dst);
            break;
        case DataType::BF16:
            run_transpose<nntile::bf16_t>(runtime, alpha, ndim, src, dst);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for tile transpose");
        default:
            throw std::runtime_error("Unsupported data type for tile transpose");
    }
}

} // namespace nntile::graph::tile_graph
