/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tile/multiply.cc
 * TileGraph multiply operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tile/multiply.hh"

#include <stdexcept>

#include <nntile/base_types.hh>
#include <nntile/graph/dtype.hh>
#include <nntile/graph/tile.hh>
#include <nntile/tile/multiply.hh>

namespace nntile::graph::tile_graph
{

namespace
{

template<typename T>
void run_multiply(
    TileGraph::Runtime& runtime,
    Scalar alpha,
    TileGraph::TileNode* x,
    TileGraph::TileNode* y,
    TileGraph::TileNode* z)
{
    auto& x_t = runtime.get_tile<T>(x);
    auto& y_t = runtime.get_tile<T>(y);
    auto& z_t = runtime.get_tile<T>(z);
    nntile::tile::multiply<T>(alpha, x_t, y_t, z_t);
}

} // namespace

void multiply(
    Scalar alpha,
    TileGraph::TileNode* x,
    TileGraph::TileNode* y,
    TileGraph::TileNode* z)
{
    if(x == nullptr || y == nullptr || z == nullptr)
    {
        throw std::invalid_argument(
            "tile multiply: x, y, z must be non-null");
    }
    if(x->graph() != y->graph() || x->graph() != z->graph())
    {
        throw std::invalid_argument(
            "tile multiply: tiles must belong to the same graph");
    }
    if(x->dtype() != y->dtype() || x->dtype() != z->dtype())
    {
        throw std::invalid_argument("tile multiply: dtype mismatch");
    }
    if(x->shape() != y->shape() || x->shape() != z->shape())
    {
        throw std::invalid_argument("tile multiply: shape mismatch");
    }
    if(x == y || x == z || y == z)
    {
        throw std::invalid_argument(
            "tile multiply: x, y, and z must be distinct tiles");
    }

    auto op = std::make_shared<TileMultiplyOp>(alpha, x, y, z);
    x->graph()->add_op(op);
}

void TileMultiplyOp::execute(TileGraph::Runtime& runtime) const
{
    DataType dtype = x->dtype();

    switch(dtype)
    {
        case DataType::FP32:
            run_multiply<nntile::fp32_t>(runtime, alpha, x, y, z);
            break;
        case DataType::FP32_FAST_TF32:
            run_multiply<nntile::fp32_fast_tf32_t>(runtime, alpha, x, y, z);
            break;
        case DataType::FP32_FAST_FP16:
            run_multiply<nntile::fp32_fast_fp16_t>(runtime, alpha, x, y, z);
            break;
        case DataType::FP32_FAST_BF16:
            run_multiply<nntile::fp32_fast_bf16_t>(runtime, alpha, x, y, z);
            break;
        case DataType::FP64:
            run_multiply<nntile::fp64_t>(runtime, alpha, x, y, z);
            break;
        case DataType::FP16:
            run_multiply<nntile::fp16_t>(runtime, alpha, x, y, z);
            break;
        case DataType::BF16:
            run_multiply<nntile::bf16_t>(runtime, alpha, x, y, z);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for tile multiply");
        default:
            throw std::runtime_error("Unsupported data type for tile multiply");
    }
}

} // namespace nntile::graph::tile_graph
