/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tile/total_sum_accum.cc
 * TileGraph total sum accum operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tile/total_sum_accum.hh"
#include <stdexcept>
#include <nntile/base_types.hh>
#include <nntile/graph/dtype.hh>
#include <nntile/graph/tile.hh>
#include <nntile/tile/total_sum_accum.hh>
namespace nntile::graph::tile_graph
{
namespace
{
template<typename T>
void run(
    TileGraph::Runtime& runtime,
    Scalar a,
    TileGraph::TileNode* lse,
    TileGraph::TileNode* s,
    TileGraph::TileNode* cl,
    TileGraph::TileNode* v,
    Index ig)
{
    nntile::tile::total_sum_accum<T>(
        a, runtime.get_tile<T>(lse), runtime.get_tile<T>(s), runtime.get_tile<nntile::int64_t>(cl), runtime.get_tile<nntile::fp32_t>(v), ig);
}
} // namespace
void total_sum_accum(Scalar a, TileGraph::TileNode* lse, TileGraph::TileNode* src, TileGraph::TileNode* labels, TileGraph::TileNode* val, Index ignore_index)
{
    if(!lse || !src || !labels || !val)
        throw std::invalid_argument("total_sum_accum");
    if(lse->graph() != src->graph() || lse->graph() != labels->graph() || lse->graph() != val->graph())
        throw std::invalid_argument("total_sum_accum");
    if(labels->dtype() != DataType::INT64 || val->dtype() != DataType::FP32)
        throw std::invalid_argument("total_sum_accum: labels INT64, val FP32");
    if(lse->dtype() != src->dtype())
        throw std::invalid_argument("total_sum_accum");
    lse->graph()->add_op(
        std::make_shared<TileTotalSumAccumOp>(a, lse, src, labels, val, ignore_index));
}
void TileTotalSumAccumOp::execute(TileGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(src);
    switch(dtype)
    {
        case DataType::FP32:
            run<nntile::fp32_t>(runtime, alpha, logsumexp, src, class_labels, val, ignore_index);
            break;
        case DataType::FP32_FAST_TF32:
            run<nntile::fp32_fast_tf32_t>(runtime, alpha, logsumexp, src, class_labels, val, ignore_index);
            break;
        case DataType::FP32_FAST_FP16:
            run<nntile::fp32_fast_fp16_t>(runtime, alpha, logsumexp, src, class_labels, val, ignore_index);
            break;
        case DataType::FP32_FAST_BF16:
            run<nntile::fp32_fast_bf16_t>(runtime, alpha, logsumexp, src, class_labels, val, ignore_index);
            break;
        case DataType::FP64:
            run<nntile::fp64_t>(runtime, alpha, logsumexp, src, class_labels, val, ignore_index);
            break;
        case DataType::FP16:
            run<nntile::fp16_t>(runtime, alpha, logsumexp, src, class_labels, val, ignore_index);
            break;
        case DataType::BF16:
            run<nntile::bf16_t>(runtime, alpha, logsumexp, src, class_labels, val, ignore_index);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error("total_sum_accum");
        default:
            throw std::runtime_error("total_sum_accum");
    }
}
} // namespace nntile::graph::tile_graph
