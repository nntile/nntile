#include <nntile/common.hh>
/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file src/tile/ops/swap_two_axes.cc
 * TileGraph swap_two_axes operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/tile/ops/swap_two_axes.hh"

#include <stdexcept>

#include <nntile/core.hh>
#include <nntile/core/swap_two_axes.hh>
#include <nntile/runtime.hh>

namespace nntile::tile
{

namespace
{

template<typename T>
void run_swap_two_axes(
    Runtime &runtime,
    Index dim0,
    Index dim1,
    TileGraph::TileNode *src,
    TileGraph::TileNode *dst)
{
    auto &s = runtime.get_tile<T>(src);
    auto &d = runtime.get_tile<T>(dst);
    nntile::core::swap_two_axes<T>(
        runtime.starpu_worker_hint(),
        s,
        d,
        dim0,
        dim1);
}

} // namespace

void swap_two_axes(
    TileGraph::TileNode *src,
    TileGraph::TileNode *dst,
    Index dim0,
    Index dim1)
{
    if (src == nullptr || dst == nullptr)
    {
        throw std::invalid_argument(
            "tile swap_two_axes: src and dst must be non-null");
    }
    if (src->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "tile swap_two_axes: src and dst must belong to the same graph");
    }
    if (src->dtype() != dst->dtype())
    {
        throw std::invalid_argument("tile swap_two_axes: dtype mismatch");
    }
    if (src == dst)
    {
        throw std::invalid_argument(
            "tile swap_two_axes: src and dst must be distinct");
    }
    auto op = std::make_shared<TileSwapTwoAxesOp>(src, dst, dim0, dim1);
    src->graph()->add_op(op);
}

void TileSwapTwoAxesOp::execute(Runtime &runtime) const
{
    DataType dtype = runtime.get_dtype(src);
    switch (dtype)
    {
        case DataType::FP32:
            run_swap_two_axes<nntile::fp32_t>(
                runtime,
                dim0,
                dim1,
                src,
                dst);
            break;
        case DataType::FP32_FAST_TF32:
            run_swap_two_axes<nntile::fp32_fast_tf32_t>(
                runtime,
                dim0,
                dim1,
                src,
                dst);
            break;
        case DataType::FP32_FAST_FP16:
            run_swap_two_axes<nntile::fp32_fast_fp16_t>(
                runtime,
                dim0,
                dim1,
                src,
                dst);
            break;
        case DataType::FP32_FAST_BF16:
            run_swap_two_axes<nntile::fp32_fast_bf16_t>(
                runtime,
                dim0,
                dim1,
                src,
                dst);
            break;
        case DataType::FP64:
            run_swap_two_axes<nntile::fp64_t>(
                runtime,
                dim0,
                dim1,
                src,
                dst);
            break;
        case DataType::FP16:
            run_swap_two_axes<nntile::fp16_t>(
                runtime,
                dim0,
                dim1,
                src,
                dst);
            break;
        case DataType::BF16:
            run_swap_two_axes<nntile::bf16_t>(
                runtime,
                dim0,
                dim1,
                src,
                dst);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for tile swap_two_axes");
        default:
            throw std::runtime_error(
                "Unsupported data type for tile swap_two_axes");
    }
}

} // namespace nntile::tile
