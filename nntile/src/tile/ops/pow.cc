/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file nntile/src/tile_graph/pow.cc
 * TileGraph pow operation implementation.
 *
 * @version 2.0.0
 * */
/*!
 * INTENTIONALLY DISABLED in the NNTile 2.0 transformer-only scope.
 * Kept in-tree for future re-enablement. No product path calls this code;
 * its tests are not registered while the family is in
 * NNTILE_DISABLED_KERNELS.
 */
#include "nntile/tile/ops/pow.hh"

#include <stdexcept>

#include <nntile/base_types.hh>
#include <nntile/dtype.hh>
#include <nntile/kernel_scope.hh>
#include <nntile/core.hh>
#include <nntile/core/pow.hh>

#include <nntile/runtime.hh>

namespace nntile::tile
{

namespace
{

template<typename T>
void run_pow(
    Runtime& runtime,
    Scalar alpha,
    Scalar exp,
    TileGraph::TileNode* A)
{
    auto& t = runtime.get_tile<T>(A);
    nntile::core::pow<T>(runtime.worker_hint(), alpha, exp, t);
}

} // namespace

void pow(Scalar alpha, Scalar exp, TileGraph::TileNode* A)
{
    nntile::require_kernel_family("pow");
    if(A == nullptr)
    {
        throw std::invalid_argument("tile pow: A must be non-null");
    }
    auto op = std::make_shared<TilePowOp>(alpha, exp, A);
    A->graph()->add_op(op);
}

void TilePowOp::execute(Runtime& runtime) const
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

} // namespace nntile::tile
