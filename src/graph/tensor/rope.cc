/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/rope.cc
 * TensorGraph rope operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/rope.hh"

#include <stdexcept>
#include <utility>

#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/rope.hh"

#include "nntile/graph/tile/lowering_context.hh"
#include "nntile/graph/tile/rope.hh"
#include "nntile/graph/tensor/tensor_graph_tiling.hh"
#include "nntile/graph/tensor/tile_lowering_helpers.hh"

namespace nntile::graph::tensor
{

namespace
{

template<typename T>
void run_rope(
    TensorGraph::Runtime& runtime,
    TensorGraph::TensorNode* sin,
    TensorGraph::TensorNode* cos,
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst)
{
    auto& sin_t = runtime.get_tensor<T>(sin);
    auto& cos_t = runtime.get_tensor<T>(cos);
    auto& src_t = runtime.get_tensor<T>(src);
    auto& dst_t = runtime.get_tensor<T>(dst);
    nntile::tensor::rope<T>(sin_t, cos_t, src_t, dst_t);
}

} // namespace

TensorGraph::TensorNode* rope(
    TensorGraph::TensorNode* sin,
    TensorGraph::TensorNode* cos,
    TensorGraph::TensorNode* src,
    const std::string& output_name)
{
    if(sin == nullptr || cos == nullptr || src == nullptr)
    {
        throw std::invalid_argument(
            "rope: input tensors must be non-null");
    }
    if(sin->graph() != cos->graph() || sin->graph() != src->graph())
    {
        throw std::invalid_argument(
            "rope: input tensors must belong to the same graph");
    }
    if(sin->dtype() != cos->dtype() || sin->dtype() != src->dtype())
    {
        throw std::invalid_argument(
            "rope: input tensors must have the same dtype");
    }

    TensorGraph::TensorNode* dst = src->graph()->data(
        src->shape(),
        output_name,
        src->dtype());
    dst->set_axes(src->axes());

    rope(sin, cos, src, dst);

    return dst;
}

void rope(
    TensorGraph::TensorNode* sin,
    TensorGraph::TensorNode* cos,
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst)
{
    if(sin == nullptr || cos == nullptr || src == nullptr || dst == nullptr)
    {
        throw std::invalid_argument(
            "rope: input tensors must be non-null");
    }
    if(sin->graph() != cos->graph() || sin->graph() != src->graph() ||
       sin->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "rope: input tensors must belong to the same graph");
    }
    if(sin->dtype() != cos->dtype() || sin->dtype() != src->dtype() ||
       sin->dtype() != dst->dtype())
    {
        throw std::invalid_argument(
            "rope: input tensors must have the same dtype");
    }
    validate_same_shape_and_merge(src, dst, "rope");

    auto op = std::make_shared<TensorRopeOp>(sin, cos, src, dst);
    src->graph()->add_op(op);
}

void TensorRopeOp::execute(
    TensorGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(src);

    switch(dtype)
    {
        case DataType::FP32:
            run_rope<nntile::fp32_t>(runtime, sin, cos, src, dst);
            break;
        case DataType::FP32_FAST_TF32:
            run_rope<nntile::fp32_fast_tf32_t>(runtime, sin, cos, src, dst);
            break;
        case DataType::FP32_FAST_FP16:
            run_rope<nntile::fp32_fast_fp16_t>(runtime, sin, cos, src, dst);
            break;
        case DataType::FP32_FAST_BF16:
            run_rope<nntile::fp32_fast_bf16_t>(runtime, sin, cos, src, dst);
            break;
        case DataType::FP64:
            run_rope<nntile::fp64_t>(runtime, sin, cos, src, dst);
            break;
        case DataType::FP16:
            run_rope<nntile::fp16_t>(runtime, sin, cos, src, dst);
            break;
        case DataType::BF16:
            run_rope<nntile::bf16_t>(runtime, sin, cos, src, dst);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for rope operation");
        default:
            throw std::runtime_error("Unsupported data type for rope");
    }
}

void TensorRopeOp::lower_to_tile(const LoweringContext& ctx) const
{
    // Match nntile::tensor::rope_async (src/tensor/rope.cc).
    tile_lower::assert_same_elementwise_layout(src, dst, "ROPE src/dst");

    const TensorAxisLayout* lay_src = ctx.tiling.find(src);
    const TensorAxisLayout* lay_sin = ctx.tiling.find(sin);
    if(lay_src == nullptr || lay_sin == nullptr)
    {
        throw std::runtime_error(
            "lower_to_tile ROPE: missing tiling for src and/or sin");
    }

    const auto& tiles_sin = tile_lower::tiles_of(ctx.tile_map, sin);
    const auto& tiles_cos = tile_lower::tiles_of(ctx.tile_map, cos);
    const auto& tiles_src = tile_lower::tiles_of(ctx.tile_map, src);
    const auto& tiles_dst = tile_lower::tiles_of(ctx.tile_map, dst);

    const Index sin_ndim = sin->ndim();
    std::vector<Index> src_coord;
    std::vector<Index> sincos_coord(static_cast<size_t>(sin_ndim));

    for(Index lin = 0; lin < lay_src->grid_volume(); ++lin)
    {
        lay_src->grid_coord_from_linear(lin, src_coord);
        for(Index d = 0; d < sin_ndim; ++d)
        {
            sincos_coord[static_cast<size_t>(d)] =
                src_coord[static_cast<size_t>(d)];
        }
        const Index j = lay_sin->grid_linear(sincos_coord);
        tile_graph::rope(
            tiles_sin[static_cast<size_t>(j)],
            tiles_cos[static_cast<size_t>(j)],
            tiles_src[static_cast<size_t>(lin)],
            tiles_dst[static_cast<size_t>(lin)]);
    }
}

} // namespace nntile::graph::tensor
