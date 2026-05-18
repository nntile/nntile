/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tile/hypot_inplace.cc
 * TileGraph hypot inplace operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tile/hypot_inplace.hh"
#include <stdexcept>
#include <nntile/base_types.hh>
#include <nntile/graph/dtype.hh>
#include <nntile/graph/tile.hh>
#include <nntile/tile/hypot_inplace.hh>
namespace nntile::graph::tile_graph
{
namespace
{
template<typename T>
void run(
    TileGraph::Runtime& runtime, Scalar a, TileGraph::TileNode* s, Scalar b, TileGraph::TileNode* d)
{
    nntile::tile::hypot_inplace<T>(a, runtime.get_tile<T>(s), b, runtime.get_tile<T>(d));
}
} // namespace
void hypot_inplace(Scalar alpha, TileGraph::TileNode* src, Scalar beta, TileGraph::TileNode* dst)
{
    if(!src || !dst)
        throw std::invalid_argument("tile hypot_inplace: null");
    if(src->graph() != dst->graph())
        throw std::invalid_argument("tile hypot_inplace: same graph");
    if(src->dtype() != dst->dtype())
        throw std::invalid_argument("tile hypot_inplace: dtype");
    if(src->shape() != dst->shape())
        throw std::invalid_argument("tile hypot_inplace: shape");
    if(src == dst)
        throw std::invalid_argument("tile hypot_inplace: src and dst must be distinct");
    src->graph()->add_op(std::make_shared<TileHypotInplaceOp>(alpha, src, beta, dst));
}
void TileHypotInplaceOp::execute(TileGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(src);
    switch(dtype)
    {
        case DataType::FP32:
            run<nntile::fp32_t>(runtime, alpha, src, beta, dst);
            break;
        case DataType::FP32_FAST_TF32:
            run<nntile::fp32_fast_tf32_t>(runtime, alpha, src, beta, dst);
            break;
        case DataType::FP32_FAST_FP16:
            run<nntile::fp32_fast_fp16_t>(runtime, alpha, src, beta, dst);
            break;
        case DataType::FP32_FAST_BF16:
            run<nntile::fp32_fast_bf16_t>(runtime, alpha, src, beta, dst);
            break;
        case DataType::FP64:
            run<nntile::fp64_t>(runtime, alpha, src, beta, dst);
            break;
        case DataType::FP16:
            throw std::runtime_error("FP16 not supported for tile hypot_inplace in this build");
        case DataType::BF16:
            run<nntile::bf16_t>(runtime, alpha, src, beta, dst);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) + " not supported for hypot_inplace");
        default:
            throw std::runtime_error("Unsupported data type for hypot_inplace");
    }
}
} // namespace nntile::graph::tile_graph
