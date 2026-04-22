/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tile/pow.cc
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tile/pow.hh"

#include <stdexcept>

#include <nntile/base_types.hh>
#include <nntile/graph/dtype.hh>
#include <nntile/graph/tile.hh>
#include <nntile/tile/pow.hh>

namespace nntile::graph::tile_graph
{

namespace
{

template<typename T>
void run_pow(
    TileGraph::Runtime& runtime,
    Scalar alpha,
    Scalar exp,
    TileGraph::TileNode* A)
{
    auto& t = runtime.get_tile<T>(A);
    nntile::tile::pow<T>(alpha, exp, t);
}

} // namespace

void pow(Scalar alpha, Scalar exp, TileGraph::TileNode* A)
{
    if(A == nullptr)
    {
        throw std::invalid_argument("tile pow: A must be non-null");
    }
    auto op = std::make_shared<TilePowOp>(alpha, exp, A);
    A->graph()->add_op(op);
}

void TilePowOp::execute(TileGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(A);

    switch(dtype)
    {
        case DataType::FP32:
            run_pow<nntile::fp32_t>(runtime, alpha, exp, A);
            break;
        case DataType::FP32_FAST_TF32:
            run_pow<nntile::fp32_fast_tf32_t>(runtime, alpha, exp, A);
            break;
        case DataType::FP32_FAST_FP16:
            run_pow<nntile::fp32_fast_fp16_t>(runtime, alpha, exp, A);
            break;
        case DataType::FP32_FAST_BF16:
            run_pow<nntile::fp32_fast_bf16_t>(runtime, alpha, exp, A);
            break;
        case DataType::FP64:
            run_pow<nntile::fp64_t>(runtime, alpha, exp, A);
            break;
        case DataType::FP16:
            throw std::runtime_error(
                "FP16 data type not supported for tile pow");
        case DataType::BF16:
            run_pow<nntile::bf16_t>(runtime, alpha, exp, A);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for tile pow");
        default:
            throw std::runtime_error("Unsupported data type for tile pow");
    }
}

} // namespace nntile::graph::tile_graph
