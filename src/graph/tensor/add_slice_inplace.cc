/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/add_slice_inplace.cc
 * TensorGraph add_slice_inplace operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/add_slice_inplace.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/add_slice_inplace.hh"

#include "nntile/graph/tile/add_slice_inplace.hh"
#include "nntile/graph/tile/lowering_context.hh"
#include "nntile/graph/tensor/tensor_graph_tiling.hh"
#include "nntile/graph/tensor/tile_lowering_helpers.hh"

namespace nntile::graph::tensor
{

namespace
{

template<typename T>
void run_add_slice_inplace(
    TensorGraph::Runtime& runtime,
    Scalar alpha, Scalar beta,
    Index axis,
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst)
{
    auto& src_t = runtime.get_tensor<T>(src);
    auto& dst_t = runtime.get_tensor<T>(dst);
    nntile::tensor::add_slice_inplace<T>(alpha, src_t, beta, dst_t, axis);
}

} // namespace

void add_slice_inplace(
    Scalar alpha,
    TensorGraph::TensorNode* src,
    Scalar beta,
    TensorGraph::TensorNode* dst,
    Index axis)
{
    if(src == nullptr || dst == nullptr)
    {
        throw std::invalid_argument(
            "add_slice_inplace: input tensors must be non-null");
    }
    if(src->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "add_slice_inplace: input tensors must belong to the same graph");
    }
    if(src->dtype() != dst->dtype())
    {
        throw std::invalid_argument(
            "add_slice_inplace: input tensors must have the same dtype");
    }
    if(src == dst)
    {
        throw std::invalid_argument(
            "add_slice_inplace: src and dst must be distinct tensors");
    }
    validate_slice_shape_and_merge(src, dst, axis,
                                            "add_slice_inplace");

    auto op = std::make_shared<TensorAddSliceInplaceOp>(
        src, dst, alpha, beta, axis);
    src->graph()->add_op(op);
}

void TensorAddSliceInplaceOp::execute(
    TensorGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(src);

    switch(dtype)
    {
        case DataType::FP32:
            run_add_slice_inplace<nntile::fp32_t>(
                runtime, alpha, beta, axis, src, dst);
            break;
        case DataType::FP32_FAST_TF32:
            run_add_slice_inplace<nntile::fp32_fast_tf32_t>(
                runtime, alpha, beta, axis, src, dst);
            break;
        case DataType::FP32_FAST_FP16:
            run_add_slice_inplace<nntile::fp32_fast_fp16_t>(
                runtime, alpha, beta, axis, src, dst);
            break;
        case DataType::FP32_FAST_BF16:
            run_add_slice_inplace<nntile::fp32_fast_bf16_t>(
                runtime, alpha, beta, axis, src, dst);
            break;
        case DataType::FP64:
            run_add_slice_inplace<nntile::fp64_t>(
                runtime, alpha, beta, axis, src, dst);
            break;
        case DataType::FP16:
            run_add_slice_inplace<nntile::fp16_t>(
                runtime, alpha, beta, axis, src, dst);
            break;
        case DataType::BF16:
            run_add_slice_inplace<nntile::bf16_t>(
                runtime, alpha, beta, axis, src, dst);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for add_slice_inplace operation");
        default:
            throw std::runtime_error("Unsupported data type for add_slice_inplace");
    }
}

void TensorAddSliceInplaceOp::lower_to_tile(const LoweringContext& ctx) const
{
    // Match nntile::tensor::add_slice_inplace_async (src/tensor/add_slice_inplace.cc).
    const TensorAxisLayout* lay_s = ctx.tiling.find(src);
    const TensorAxisLayout* lay_d = ctx.tiling.find(dst);
    if(lay_s == nullptr || lay_d == nullptr)
    {
        throw std::runtime_error(
            "lower_to_tile ADD_SLICE_INPLACE: missing tiling for src and/or dst");
    }

    const auto& tiles_s = tile_lower::tiles_of(ctx.tile_map, src);
    const auto& tiles_d = tile_lower::tiles_of(ctx.tile_map, dst);

    std::vector<Index> s_coord;
    std::vector<Index> d_coord(static_cast<size_t>(dst->ndim()));

    for(Index lin_s = 0; lin_s < lay_s->grid_volume(); ++lin_s)
    {
        lay_s->grid_coord_from_linear(lin_s, s_coord);
        for(Index j = 0; j < axis; ++j)
        {
            d_coord[static_cast<size_t>(j)] =
                s_coord[static_cast<size_t>(j)];
        }
        for(Index j = axis + 1; j < dst->ndim(); ++j)
        {
            d_coord[static_cast<size_t>(j)] =
                s_coord[static_cast<size_t>(j - 1)];
        }

        const Index nseg_along_axis =
            lay_d->grid_shape()[static_cast<size_t>(axis)];
        for(Index jj = 0; jj < nseg_along_axis; ++jj)
        {
            d_coord[static_cast<size_t>(axis)] = jj;
            const Index lin_d = lay_d->grid_linear(d_coord);
            tile_graph::add_slice_inplace(
                alpha,
                tiles_s[static_cast<size_t>(lin_s)],
                beta,
                tiles_d[static_cast<size_t>(lin_d)],
                axis);
        }
    }
}

} // namespace nntile::graph::tensor
