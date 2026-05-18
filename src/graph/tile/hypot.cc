/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tile/hypot.cc
 * TileGraph hypot operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tile/hypot.hh"
#include <stdexcept>
#include <nntile/base_types.hh>
#include <nntile/graph/dtype.hh>
#include <nntile/graph/tile.hh>
#include <nntile/tile/hypot.hh>
namespace nntile::graph::tile_graph
{
namespace
{
template<typename T>
void run_hy(
    TileGraph::Runtime& runtime, Scalar a, TileGraph::TileNode* s1, Scalar b, TileGraph::TileNode* s2, TileGraph::TileNode* d)
{
    nntile::tile::hypot<T>(a, runtime.get_tile<T>(s1), b, runtime.get_tile<T>(s2), runtime.get_tile<T>(d));
}
} // namespace
void hypot(
    Scalar alpha, TileGraph::TileNode* src1, Scalar beta, TileGraph::TileNode* src2, TileGraph::TileNode* dst)
{
    if(!src1 || !src2 || !dst)
        throw std::invalid_argument("tile hypot: null");
    if(src1->graph() != src2->graph() || src1->graph() != dst->graph())
        throw std::invalid_argument("tile hypot: same graph");
    if(src1->dtype() != src2->dtype() || src1->dtype() != dst->dtype())
        throw std::invalid_argument("tile hypot: dtype");
    if(src1->shape() != src2->shape() || src1->shape() != dst->shape())
        throw std::invalid_argument("tile hypot: shape");
    if(src1 == src2 || src1 == dst || src2 == dst)
        throw std::invalid_argument("tile hypot: distinct");
    src1->graph()->add_op(std::make_shared<TileHypotOp>(alpha, src1, beta, src2, dst));
}
void TileHypotOp::execute(TileGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(src1);
    switch(dtype)
    {
        case DataType::FP32:
            run_hy<nntile::fp32_t>(runtime, alpha, src1, beta, src2, dst);
            break;
        case DataType::FP32_FAST_TF32:
            run_hy<nntile::fp32_fast_tf32_t>(runtime, alpha, src1, beta, src2, dst);
            break;
        case DataType::FP32_FAST_FP16:
            run_hy<nntile::fp32_fast_fp16_t>(runtime, alpha, src1, beta, src2, dst);
            break;
        case DataType::FP32_FAST_BF16:
            run_hy<nntile::fp32_fast_bf16_t>(runtime, alpha, src1, beta, src2, dst);
            break;
        case DataType::FP64:
            run_hy<nntile::fp64_t>(runtime, alpha, src1, beta, src2, dst);
            break;
        case DataType::FP16:
            run_hy<nntile::fp16_t>(runtime, alpha, src1, beta, src2, dst);
            break;
        case DataType::BF16:
            run_hy<nntile::bf16_t>(runtime, alpha, src1, beta, src2, dst);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) + " not supported for tile hypot");
        default:
            throw std::runtime_error("Unsupported data type for tile hypot");
    }
}
} // namespace nntile::graph::tile_graph
