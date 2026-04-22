/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tile/relu_backward.cc
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tile/relu_backward.hh"

#include <stdexcept>

#include <nntile/base_types.hh>
#include <nntile/graph/dtype.hh>
#include <nntile/graph/tile.hh>
#include <nntile/tile/relu_backward.hh>

namespace nntile::graph::tile_graph
{

namespace
{

template<typename T>
void run_relu_backward(
    TileGraph::Runtime& runtime,
    TileGraph::TileNode* x,
    TileGraph::TileNode* dy,
    TileGraph::TileNode* dx)
{
    auto& x_t = runtime.get_tile<T>(x);
    auto& dy_t = runtime.get_tile<T>(dy);
    auto& dx_t = runtime.get_tile<T>(dx);
    nntile::tile::relu_backward<T>(x_t, dy_t, dx_t);
}

} // namespace

void relu_backward(
    TileGraph::TileNode* x, TileGraph::TileNode* dy, TileGraph::TileNode* dx)
{
    if(x == nullptr || dy == nullptr || dx == nullptr)
    {
        throw std::invalid_argument(
            "tile relu_backward: x, dy, dx must be non-null");
    }
    if(x->graph() != dy->graph() || x->graph() != dx->graph())
    {
        throw std::invalid_argument(
            "tile relu_backward: operands must belong to the same graph");
    }
    if(x->dtype() != dy->dtype() || x->dtype() != dx->dtype())
    {
        throw std::invalid_argument("tile relu_backward: dtype mismatch");
    }
    if(x->shape() != dy->shape() || x->shape() != dx->shape())
    {
        throw std::invalid_argument("tile relu_backward: shape mismatch");
    }

    auto op = std::make_shared<TileReluBackwardOp>(x, dy, dx);
    x->graph()->add_op(op);
}

void TileReluBackwardOp::execute(TileGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(x);

    switch(dtype)
    {
        case DataType::FP32:
            run_relu_backward<nntile::fp32_t>(runtime, x, dy, dx);
            break;
        case DataType::FP32_FAST_TF32:
            run_relu_backward<nntile::fp32_fast_tf32_t>(runtime, x, dy, dx);
            break;
        case DataType::FP32_FAST_FP16:
            run_relu_backward<nntile::fp32_fast_fp16_t>(runtime, x, dy, dx);
            break;
        case DataType::FP32_FAST_BF16:
            run_relu_backward<nntile::fp32_fast_bf16_t>(runtime, x, dy, dx);
            break;
        case DataType::FP64:
            run_relu_backward<nntile::fp64_t>(runtime, x, dy, dx);
            break;
        case DataType::FP16:
            throw std::runtime_error(
                "FP16 data type not supported for tile relu_backward");
        case DataType::BF16:
            run_relu_backward<nntile::bf16_t>(runtime, x, dy, dx);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for tile relu_backward");
        default:
            throw std::runtime_error(
                "Unsupported data type for tile relu_backward");
    }
}

} // namespace nntile::graph::tile_graph
