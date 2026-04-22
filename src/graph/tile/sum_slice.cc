/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tile/sum_slice.cc
 * TileGraph sum slice operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tile/sum_slice.hh"
#include <stdexcept>
#include <nntile/base_types.hh>
#include <nntile/graph/dtype.hh>
#include <nntile/graph/tile.hh>
#include <nntile/tile/sum_slice.hh>
namespace nntile::graph::tile_graph
{
namespace
{
template<typename T>
void run(
    TileGraph::Runtime& runtime, TileGraph::TileNode* s, TileGraph::TileNode* d, Scalar a, Scalar b, Index ax, int r)
{
    nntile::tile::sum_slice<T>(a, runtime.get_tile<T>(s), b, runtime.get_tile<T>(d), ax, r);
}
} // namespace
void sum_slice(
    Scalar alpha, TileGraph::TileNode* src, Scalar beta, TileGraph::TileNode* dst, Index axis, int redux)
{
    if(!src || !dst)
        throw std::invalid_argument("tile sum_slice");
    if(src->graph() != dst->graph() || src->dtype() != dst->dtype() || src == dst)
        throw std::invalid_argument("tile sum_slice: invalid");
    src->graph()->add_op(
        std::make_shared<TileSumSliceOp>(alpha, src, beta, dst, axis, redux));
}
void TileSumSliceOp::execute(TileGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(src);
    switch(dtype)
    {
        case DataType::FP32:
            run<nntile::fp32_t>(runtime, src, dst, alpha, beta, axis, redux);
            break;
        case DataType::FP32_FAST_TF32:
            run<nntile::fp32_fast_tf32_t>(runtime, src, dst, alpha, beta, axis, redux);
            break;
        case DataType::FP32_FAST_FP16:
            run<nntile::fp32_fast_fp16_t>(runtime, src, dst, alpha, beta, axis, redux);
            break;
        case DataType::FP32_FAST_BF16:
            run<nntile::fp32_fast_bf16_t>(runtime, src, dst, alpha, beta, axis, redux);
            break;
        case DataType::FP64:
            run<nntile::fp64_t>(runtime, src, dst, alpha, beta, axis, redux);
            break;
        case DataType::FP16:
            run<nntile::fp16_t>(runtime, src, dst, alpha, beta, axis, redux);
            break;
        case DataType::BF16:
            run<nntile::bf16_t>(runtime, src, dst, alpha, beta, axis, redux);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) + " not supported for sum_slice");
        default:
            throw std::runtime_error("Unsupported data type for sum_slice");
    }
}
} // namespace nntile::graph::tile_graph
