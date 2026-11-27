/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file nntile/src/tile_graph/sqrt.cc
 * TileGraph sqrt operation implementation.
 *
 * @version 1.1.0
 * */
/*!
 * INTENTIONALLY DISABLED in the NNTile 2.0 transformer-only scope.
 * Kept in-tree for future re-enablement. No product path calls this code;
 * its tests are not registered while the family is in
 * NNTILE_DISABLED_KERNELS.
 */
#include "nntile/tile/ops/sqrt.hh"

#include <stdexcept>

#include <nntile/base_types.hh>
#include <nntile/dtype.hh>
#include <nntile/kernel_scope.hh>
#include <nntile/core.hh>
#include <nntile/core/sqrt.hh>

#include <nntile/runtime.hh>

namespace nntile::tile
{

namespace
{

template<typename T>
void run_sqrt(
    Runtime& runtime,
    TileGraph::TileNode* src,
    TileGraph::TileNode* dst)
{
    auto& s = runtime.get_tile<T>(src);
    auto& d = runtime.get_tile<T>(dst);
    nntile::core::sqrt<T>(runtime.worker_hint(), s, d);
}

} // namespace

void sqrt(TileGraph::TileNode* src, TileGraph::TileNode* dst)
{
    nntile::require_kernel_family("sqrt");
    if(src == nullptr || dst == nullptr)
    {
        throw std::invalid_argument("tile sqrt: src and dst must be non-null");
    }
    if(src->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "tile sqrt: src and dst must belong to the same graph");
    }
    if(src->dtype() != dst->dtype())
    {
        throw std::invalid_argument("tile sqrt: dtype mismatch");
    }
    if(src->shape() != dst->shape())
    {
        throw std::invalid_argument("tile sqrt: shape mismatch");
    }

    auto op = std::make_shared<TileSqrtOp>(src, dst);
    src->graph()->add_op(op);
}

void TileSqrtOp::execute(Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(src);

    switch(dtype)
    {
        case DataType::FP32:
            run_sqrt<nntile::fp32_t>(runtime, src, dst);
            break;
        case DataType::FP32_FAST_TF32:
            run_sqrt<nntile::fp32_fast_tf32_t>(runtime, src, dst);
            break;
        case DataType::FP32_FAST_FP16:
            run_sqrt<nntile::fp32_fast_fp16_t>(runtime, src, dst);
            break;
        case DataType::FP32_FAST_BF16:
            run_sqrt<nntile::fp32_fast_bf16_t>(runtime, src, dst);
            break;
        case DataType::FP64:
            run_sqrt<nntile::fp64_t>(runtime, src, dst);
            break;
        case DataType::FP16:
            throw std::runtime_error(
                "FP16 data type not supported for tile sqrt");
        case DataType::BF16:
            run_sqrt<nntile::bf16_t>(runtime, src, dst);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for tile sqrt");
        default:
            throw std::runtime_error("Unsupported data type for tile sqrt");
    }
}

} // namespace nntile::tile
