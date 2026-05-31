/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/tile_graph/silu_inplace.cc
 * TileGraph silu inplace operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/tile/ops/silu_inplace.hh"

#include <stdexcept>

#include <nntile/base_types.hh>
#include <nntile/dtype.hh>
#include <nntile/core.hh>
#include <nntile/core/silu_inplace.hh>

#include <nntile/runtime.hh>

namespace nntile::tile
{

namespace
{

template<typename T>
void run_silu_inplace(Runtime& runtime, TileGraph::TileNode* d)
{
    auto& t = runtime.get_tile<T>(d);
    nntile::core::silu_inplace<T>(runtime.starpu_worker_hint(), t);
}

} // namespace

void silu_inplace(TileGraph::TileNode* dst)
{
    if(dst == nullptr)
    {
        throw std::invalid_argument("tile silu_inplace: dst must be non-null");
    }
    auto op = std::make_shared<TileSiluInplaceOp>(dst);
    dst->graph()->add_op(op);
}

void TileSiluInplaceOp::execute(Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(dst);

    switch(dtype)
    {
        case DataType::FP32:
            run_silu_inplace<nntile::fp32_t>(runtime, dst);
            break;
        case DataType::FP32_FAST_TF32:
            run_silu_inplace<nntile::fp32_fast_tf32_t>(runtime, dst);
            break;
        case DataType::FP32_FAST_FP16:
            run_silu_inplace<nntile::fp32_fast_fp16_t>(runtime, dst);
            break;
        case DataType::FP32_FAST_BF16:
            run_silu_inplace<nntile::fp32_fast_bf16_t>(runtime, dst);
            break;
        case DataType::FP64:
            run_silu_inplace<nntile::fp64_t>(runtime, dst);
            break;
        case DataType::FP16:
            run_silu_inplace<nntile::fp16_t>(runtime, dst);
            break;
        case DataType::BF16:
            run_silu_inplace<nntile::bf16_t>(runtime, dst);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for tile silu_inplace");
        default:
            throw std::runtime_error(
                "Unsupported data type for tile silu_inplace");
    }
}

} // namespace nntile::tile
