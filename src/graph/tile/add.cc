/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tile/add.cc
 * TileGraph add operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tile/add.hh"

#include <stdexcept>
#include <utility>

#include <nntile/base_types.hh>
#include <nntile/graph/tile.hh>
#include <nntile/tile/add.hh>

namespace nntile::graph::tile_graph
{

namespace
{

template<typename T>
void run_add(
    TileGraph::Runtime& runtime,
    Scalar alpha,
    Scalar beta,
    TileGraph::TileNode* x,
    TileGraph::TileNode* y,
    TileGraph::TileNode* z)
{
    auto& x_t = runtime.get_tile<T>(x);
    auto& y_t = runtime.get_tile<T>(y);
    auto& z_t = runtime.get_tile<T>(z);
    nntile::tile::add<T>(alpha, x_t, beta, y_t, z_t);
}

} // namespace

TileGraph::TileNode* add(
    Scalar alpha,
    TileGraph::TileNode* x,
    Scalar beta,
    TileGraph::TileNode* y,
    const std::string& output_name)
{
    if(x == nullptr || y == nullptr)
    {
        throw std::invalid_argument("tile add: input tiles must be non-null");
    }
    if(x->graph() != y->graph())
    {
        throw std::invalid_argument(
            "tile add: input tiles must belong to the same graph");
    }
    if(x->dtype() != y->dtype())
    {
        throw std::invalid_argument(
            "tile add: input tiles must have the same dtype");
    }
    if(x->shape() != y->shape())
    {
        throw std::invalid_argument(
            "tile add: input tiles must have the same shape");
    }

    std::vector<Index> output_shape = x->shape();
    TileGraph::TileNode* output = x->graph()->data(
        std::move(output_shape),
        output_name,
        x->dtype());

    add(alpha, x, beta, y, output);

    return output;
}

void add(
    Scalar alpha,
    TileGraph::TileNode* x,
    Scalar beta,
    TileGraph::TileNode* y,
    TileGraph::TileNode* z)
{
    if(x == nullptr || y == nullptr || z == nullptr)
    {
        throw std::invalid_argument("tile add: input tiles must be non-null");
    }
    if(x->graph() != y->graph() || x->graph() != z->graph())
    {
        throw std::invalid_argument(
            "tile add: input tiles must belong to the same graph");
    }
    if(x->dtype() != y->dtype() || x->dtype() != z->dtype())
    {
        throw std::invalid_argument(
            "tile add: input tiles must have the same dtype");
    }
    if(x->shape() != y->shape() || x->shape() != z->shape())
    {
        throw std::invalid_argument(
            "tile add: input tiles must have the same shape");
    }
    if(x == y || x == z || y == z)
    {
        throw std::invalid_argument(
            "tile add: x, y, and z must be distinct tiles");
    }

    auto op = std::make_shared<TileAddOp>(x, y, z, alpha, beta);
    x->graph()->add_op(op);
}

void TileAddOp::execute(TileGraph::Runtime& runtime) const
{
    DataType dtype = x->dtype();

    switch(dtype)
    {
        case DataType::FP32:
            run_add<nntile::fp32_t>(runtime, alpha, beta, x, y, z);
            break;
        case DataType::FP32_FAST_TF32:
            run_add<nntile::fp32_fast_tf32_t>(runtime, alpha, beta, x, y, z);
            break;
        case DataType::FP32_FAST_FP16:
            run_add<nntile::fp32_fast_fp16_t>(runtime, alpha, beta, x, y, z);
            break;
        case DataType::FP32_FAST_BF16:
            run_add<nntile::fp32_fast_bf16_t>(runtime, alpha, beta, x, y, z);
            break;
        case DataType::FP64:
            run_add<nntile::fp64_t>(runtime, alpha, beta, x, y, z);
            break;
        case DataType::FP16:
            run_add<nntile::fp16_t>(runtime, alpha, beta, x, y, z);
            break;
        case DataType::BF16:
            run_add<nntile::bf16_t>(runtime, alpha, beta, x, y, z);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for tile add operation");
        default:
            throw std::runtime_error("Unsupported data type for tile add");
    }
}

} // namespace nntile::graph::tile_graph
