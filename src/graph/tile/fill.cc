/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tile/fill.cc
 * TileGraph fill operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tile/fill.hh"

#include <stdexcept>

#include <nntile/base_types.hh>
#include <nntile/graph/dtype.hh>
#include <nntile/graph/tile.hh>
#include <nntile/tile/fill.hh>

namespace nntile::graph::tile_graph
{

namespace
{

template<typename T>
void run_fill(
    TileGraph::Runtime& runtime,
    Scalar val,
    TileGraph::TileNode* x)
{
    auto& x_t = runtime.get_tile<T>(x);
    nntile::tile::fill<T>(val, x_t);
}

} // namespace

void fill(Scalar val, TileGraph::TileNode* x)
{
    if(x == nullptr)
    {
        throw std::invalid_argument("tile fill: input tile must be non-null");
    }

    auto op = std::make_shared<TileFillOp>(x, val);
    x->graph()->add_op(op);
}

void TileFillOp::execute(
    TileGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(x);

    switch(dtype)
    {
        case DataType::FP32:
            run_fill<nntile::fp32_t>(runtime, val, x);
            break;
        case DataType::FP32_FAST_TF32:
            run_fill<nntile::fp32_fast_tf32_t>(runtime, val, x);
            break;
        case DataType::FP32_FAST_FP16:
            run_fill<nntile::fp32_fast_fp16_t>(runtime, val, x);
            break;
        case DataType::FP32_FAST_BF16:
            run_fill<nntile::fp32_fast_bf16_t>(runtime, val, x);
            break;
        case DataType::FP64:
            run_fill<nntile::fp64_t>(runtime, val, x);
            break;
        case DataType::FP16:
            run_fill<nntile::fp16_t>(runtime, val, x);
            break;
        case DataType::BF16:
            run_fill<nntile::bf16_t>(runtime, val, x);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                "INT64/BOOL data type not supported for tile fill operation");
        default:
            throw std::runtime_error("Unsupported data type for tile fill");
    }
}

} // namespace nntile::graph::tile_graph
