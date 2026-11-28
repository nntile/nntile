/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file nntile/src/tile_graph/sqrt_inplace.cc
 * TileGraph sqrt inplace operation implementation.
 *
 * @version 2.0.0
 * */
/*!
 * INTENTIONALLY DISABLED in the NNTile 2.0 transformer-only scope.
 * Kept in-tree for future re-enablement. No product path calls this code;
 * its tests are not registered while the family is in
 * NNTILE_DISABLED_KERNELS.
 */
#include "nntile/tile/ops/sqrt_inplace.hh"
#include <stdexcept>
#include <nntile/base_types.hh>
#include <nntile/dtype.hh>
#include <nntile/kernel_scope.hh>
#include <nntile/core.hh>
#include <nntile/core/sqrt_inplace.hh>

#include <nntile/runtime.hh>

namespace nntile::tile
{
namespace
{
template<typename T>
void run(Runtime& runtime, TileGraph::TileNode* d)
{
    auto& t = runtime.get_tile<T>(d);
    nntile::core::sqrt_inplace<T>(runtime.worker_hint(), t);
}
} // namespace
void sqrt_inplace(TileGraph::TileNode* dst)
{
    nntile::require_kernel_family("sqrt_inplace");
    if(dst == nullptr)
    {
        throw std::invalid_argument("tile sqrt_inplace: dst must be non-null");
    }
    dst->graph()->add_op(std::make_shared<TileSqrtInplaceOp>(dst));
}
void TileSqrtInplaceOp::execute(Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(dst);
    switch(dtype)
    {
        case DataType::FP32:
            run<nntile::fp32_t>(runtime, dst);
            break;
        case DataType::FP32_FAST_TF32:
            run<nntile::fp32_fast_tf32_t>(runtime, dst);
            break;
        case DataType::FP32_FAST_FP16:
            run<nntile::fp32_fast_fp16_t>(runtime, dst);
            break;
        case DataType::FP32_FAST_BF16:
            run<nntile::fp32_fast_bf16_t>(runtime, dst);
            break;
        case DataType::FP64:
            run<nntile::fp64_t>(runtime, dst);
            break;
        case DataType::FP16:
            throw std::runtime_error("FP16 not supported for tile sqrt_inplace in this build");
        case DataType::BF16:
            run<nntile::bf16_t>(runtime, dst);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for tile sqrt_inplace");
        default:
            throw std::runtime_error("Unsupported data type for tile sqrt_inplace");
    }
}
} // namespace nntile::tile
