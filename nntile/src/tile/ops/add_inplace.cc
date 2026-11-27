/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/tile_graph/add_inplace.cc
 * TileGraph add_inplace operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/tile/ops/add_inplace.hh"

#include <stdexcept>

#include <nntile/base_types.hh>
#include <nntile/core.hh>
#include <nntile/core/add_inplace.hh>

#include <nntile/runtime.hh>

namespace nntile::tile
{

namespace
{

template<typename T>
void run_add_inplace(
    Runtime& runtime,
    Scalar alpha, Scalar beta,
    TileGraph::TileNode* x,
    TileGraph::TileNode* y)
{
    auto& x_t = runtime.get_tile<T>(x);
    auto& y_t = runtime.get_tile<T>(y);
    nntile::core::add_inplace<T>(runtime.starpu_worker_hint(), alpha, x_t, beta, y_t);
}

} // namespace

void add_inplace(
    Scalar alpha,
    TileGraph::TileNode* x,
    Scalar beta,
    TileGraph::TileNode* y)
{
    if(x == nullptr || y == nullptr)
    {
        throw std::invalid_argument(
            "tile add_inplace: input tiles must be non-null");
    }
    if(x->graph() != y->graph())
    {
        throw std::invalid_argument(
            "tile add_inplace: input tiles must belong to the same graph");
    }
    if(x->dtype() != y->dtype())
    {
        throw std::invalid_argument(
            "tile add_inplace: input tiles must have the same dtype");
    }
    if(x->shape() != y->shape())
    {
        throw std::invalid_argument(
            "tile add_inplace: input tiles must have the same shape");
    }
    if(x == y)
    {
        throw std::invalid_argument(
            "tile add_inplace: x and y must be distinct tiles");
    }

    auto op = std::make_shared<TileAddInplaceOp>(x, y, alpha, beta);
    x->graph()->add_op(op);
}

void TileAddInplaceOp::execute(
    Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(x);

    switch(dtype)
    {
        case DataType::FP32:
            run_add_inplace<nntile::fp32_t>(runtime, alpha, beta, x, y);
            break;
        case DataType::FP32_FAST_TF32:
            run_add_inplace<nntile::fp32_fast_tf32_t>(runtime, alpha, beta, x, y);
            break;
        case DataType::FP32_FAST_FP16:
            run_add_inplace<nntile::fp32_fast_fp16_t>(runtime, alpha, beta, x, y);
            break;
        case DataType::FP32_FAST_BF16:
            run_add_inplace<nntile::fp32_fast_bf16_t>(runtime, alpha, beta, x, y);
            break;
        case DataType::FP64:
            run_add_inplace<nntile::fp64_t>(runtime, alpha, beta, x, y);
            break;
        case DataType::FP16:
            run_add_inplace<nntile::fp16_t>(runtime, alpha, beta, x, y);
            break;
        case DataType::BF16:
            run_add_inplace<nntile::bf16_t>(runtime, alpha, beta, x, y);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for tile add_inplace operation");
        default:
            throw std::runtime_error("Unsupported data type for tile add_inplace");
    }
}

} // namespace nntile::tile
