/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tile/softmax.cc
 * TileGraph softmax operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tile/softmax.hh"
#include <stdexcept>
#include <nntile/base_types.hh>
#include <nntile/graph/dtype.hh>
#include <nntile/graph/tile.hh>
#include <nntile/tile/softmax.hh>
namespace nntile::graph::tile_graph
{
namespace
{
template<typename T>
void run_sm(
    TileGraph::Runtime& runtime, TileGraph::TileNode* m, TileGraph::TileNode* s, Scalar a, TileGraph::TileNode* d, Index ax)
{
    nntile::tile::softmax<T>(runtime.get_tile<T>(m), runtime.get_tile<T>(s), a, runtime.get_tile<T>(d), ax);
}
} // namespace
void softmax(
    TileGraph::TileNode* maxsumexp_n, TileGraph::TileNode* src, Scalar alpha, TileGraph::TileNode* dst, Index axis)
{
    if(!maxsumexp_n || !src || !dst)
        throw std::invalid_argument("tile softmax: null");
    if(maxsumexp_n->graph() != src->graph() || maxsumexp_n->graph() != dst->graph())
        throw std::invalid_argument("tile softmax: same graph");
    if(maxsumexp_n->dtype() != src->dtype() || maxsumexp_n->dtype() != dst->dtype())
        throw std::invalid_argument("tile softmax: dtype");
    if(maxsumexp_n == src || maxsumexp_n == dst || src == dst)
        throw std::invalid_argument("tile softmax: distinct");
    if(src->shape() != dst->shape())
        throw std::invalid_argument("tile softmax: src and dst must match shape");
    // maxsumexp vs. src shape rules follow nntile::tile::softmax (see
    // src/tile/softmax.cc)
    maxsumexp_n->graph()->add_op(
        std::make_shared<TileSoftmaxOp>(maxsumexp_n, src, alpha, dst, axis));
}
void TileSoftmaxOp::execute(TileGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(src);
    switch(dtype)
    {
        case DataType::FP32:
            run_sm<nntile::fp32_t>(runtime, maxsumexp, src, alpha, dst, axis);
            break;
        case DataType::FP32_FAST_TF32:
            run_sm<nntile::fp32_fast_tf32_t>(runtime, maxsumexp, src, alpha, dst, axis);
            break;
        case DataType::FP32_FAST_FP16:
            run_sm<nntile::fp32_fast_fp16_t>(runtime, maxsumexp, src, alpha, dst, axis);
            break;
        case DataType::FP32_FAST_BF16:
            run_sm<nntile::fp32_fast_bf16_t>(runtime, maxsumexp, src, alpha, dst, axis);
            break;
        case DataType::FP64:
            run_sm<nntile::fp64_t>(runtime, maxsumexp, src, alpha, dst, axis);
            break;
        case DataType::FP16:
            run_sm<nntile::fp16_t>(runtime, maxsumexp, src, alpha, dst, axis);
            break;
        case DataType::BF16:
            run_sm<nntile::bf16_t>(runtime, maxsumexp, src, alpha, dst, axis);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) + " not supported for softmax");
        default:
            throw std::runtime_error("Unsupported data type for softmax");
    }
}
} // namespace nntile::graph::tile_graph
