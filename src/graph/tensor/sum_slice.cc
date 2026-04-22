/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/sum_slice.cc
 * TensorGraph sum_slice operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/sum_slice.hh"

#include <stdexcept>
#include <utility>
#include <vector>

#include "nntile/base_types.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/sum_slice.hh"

#include "nntile/graph/tensor/tensor_graph_tiling.hh"
#include "nntile/graph/tensor/tile_lowering_helpers.hh"
#include "nntile/graph/tile/lowering_context.hh"
#include "nntile/graph/tile/sum_slice.hh"

namespace nntile::graph::tensor
{

namespace
{

std::vector<Index> sum_slice_output_shape(
    const std::vector<Index>& src_shape,
    Index axis)
{
    std::vector<Index> out_shape;
    out_shape.reserve(src_shape.size() - 1);
    for(Index i = 0; i < src_shape.size(); ++i)
    {
        if(i != axis)
        {
            out_shape.push_back(src_shape[i]);
        }
    }
    return out_shape;
}

template<typename T>
void run_sum_slice(
    TensorGraph::Runtime& runtime,
    Scalar alpha, Scalar beta,
    Index axis, int redux,
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst)
{
    auto& src_t = runtime.get_tensor<T>(src);
    auto& dst_t = runtime.get_tensor<T>(dst);
    nntile::tensor::sum_slice<T>(alpha, src_t, beta, dst_t, axis, redux);
}

} // namespace

TensorGraph::TensorNode* sum_slice(
    TensorGraph::TensorNode* src,
    const std::string& output_name,
    Index axis,
    int redux,
    Scalar alpha,
    Scalar beta)
{
    if(src == nullptr)
    {
        throw std::invalid_argument(
            "sum_slice: input tensor must be non-null");
    }
    if(axis < 0 || axis >= src->ndim())
    {
        throw std::invalid_argument(
            "sum_slice: axis out of range");
    }

    std::vector<Index> output_shape =
        sum_slice_output_shape(src->shape(), axis);
    TensorGraph::TensorNode* output = src->graph()->data(
        std::move(output_shape),
        output_name,
        src->dtype());

    validate_slice_shape_and_merge(output, src, axis, "sum_slice");

    auto op = std::make_shared<TensorSumSliceOp>(
        src, output, axis, redux, alpha, beta);
    src->graph()->add_op(op);

    return output;
}

void sum_slice(
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst,
    Index axis,
    int redux,
    Scalar alpha,
    Scalar beta)
{
    if(src == nullptr || dst == nullptr)
    {
        throw std::invalid_argument(
            "sum_slice: input tensors must be non-null");
    }
    if(src->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "sum_slice: input tensors must belong to the same graph");
    }
    if(src->dtype() != dst->dtype())
    {
        throw std::invalid_argument(
            "sum_slice: input tensors must have the same dtype");
    }
    if(axis < 0 || axis >= src->ndim())
    {
        throw std::invalid_argument(
            "sum_slice: axis out of range");
    }
    if(src == dst)
    {
        throw std::invalid_argument(
            "sum_slice: src and dst must be distinct tensors");
    }
    validate_slice_shape_and_merge(dst, src, axis, "sum_slice");

    auto op = std::make_shared<TensorSumSliceOp>(
        src, dst, axis, redux, alpha, beta);
    src->graph()->add_op(op);
}

void TensorSumSliceOp::execute(
    TensorGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(src);

    switch(dtype)
    {
        case DataType::FP32:
            run_sum_slice<nntile::fp32_t>(
                runtime, alpha, beta, axis, redux, src, dst);
            break;
        case DataType::FP32_FAST_TF32:
            run_sum_slice<nntile::fp32_fast_tf32_t>(
                runtime, alpha, beta, axis, redux, src, dst);
            break;
        case DataType::FP32_FAST_FP16:
            run_sum_slice<nntile::fp32_fast_fp16_t>(
                runtime, alpha, beta, axis, redux, src, dst);
            break;
        case DataType::FP32_FAST_BF16:
            run_sum_slice<nntile::fp32_fast_bf16_t>(
                runtime, alpha, beta, axis, redux, src, dst);
            break;
        case DataType::FP64:
            run_sum_slice<nntile::fp64_t>(
                runtime, alpha, beta, axis, redux, src, dst);
            break;
        case DataType::FP16:
            run_sum_slice<nntile::fp16_t>(
                runtime, alpha, beta, axis, redux, src, dst);
            break;
        case DataType::BF16:
            run_sum_slice<nntile::bf16_t>(
                runtime, alpha, beta, axis, redux, src, dst);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for sum_slice operation");
        default:
            throw std::runtime_error("Unsupported data type for sum_slice");
    }
}

void TensorSumSliceOp::lower_to_tile(const LoweringContext& ctx) const
{
    // Match nntile::tensor::sum_slice_async (src/tensor/sum_slice.cc).
    const TensorAxisLayout* lay_s = ctx.tiling.find(src);
    const TensorAxisLayout* lay_d = ctx.tiling.find(dst);
    if(lay_s == nullptr || lay_d == nullptr)
    {
        throw std::runtime_error(
            "lower_to_tile SUM_SLICE: missing tiling for src and/or dst");
    }

    const auto& tiles_s = tile_lower::tiles_of(ctx.tile_map, src);
    const auto& tiles_d = tile_lower::tiles_of(ctx.tile_map, dst);

    std::vector<Index> dst_coord;
    std::vector<Index> s_coord(static_cast<size_t>(src->ndim()));

    for(Index lin_d = 0; lin_d < lay_d->grid_volume(); ++lin_d)
    {
        lay_d->grid_coord_from_linear(lin_d, dst_coord);
        TileGraph::TileNode* dst_tile = tiles_d[static_cast<size_t>(lin_d)];

        for(Index j = 0, k = 0; j < src->ndim(); ++j)
        {
            if(j == axis)
            {
                continue;
            }
            s_coord[static_cast<size_t>(j)] = dst_coord[static_cast<size_t>(k)];
            ++k;
        }

        const Index nseg_along_axis =
            lay_s->grid_shape()[static_cast<size_t>(axis)];

        s_coord[static_cast<size_t>(axis)] = 0;
        Index lin_s0 = lay_s->grid_linear(s_coord);
        tile_graph::sum_slice(
            alpha,
            tiles_s[static_cast<size_t>(lin_s0)],
            beta,
            dst_tile,
            axis,
            redux);

        for(Index jj = 1; jj < nseg_along_axis; ++jj)
        {
            s_coord[static_cast<size_t>(axis)] = jj;
            const Index lin_s = lay_s->grid_linear(s_coord);
            tile_graph::sum_slice(
                alpha,
                tiles_s[static_cast<size_t>(lin_s)],
                Scalar(1.0),
                dst_tile,
                axis,
                redux);
        }
    }
}

} // namespace nntile::graph::tensor
