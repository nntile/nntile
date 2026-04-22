/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/multiply_fiber.cc
 * TensorGraph multiply_fiber operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/multiply_fiber.hh"

#include <stdexcept>
#include <utility>

#include "nntile/base_types.hh"
#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/multiply_fiber.hh"

#include "nntile/graph/tile/lowering_context.hh"
#include "nntile/graph/tensor/tensor_graph_tiling.hh"
#include "nntile/graph/tensor/tile_lowering_helpers.hh"
#include "nntile/graph/tile/multiply_fiber.hh"

namespace nntile::graph::tensor
{

namespace
{

template<typename T>
void run_multiply_fiber(
    TensorGraph::Runtime& runtime,
    Scalar alpha, Index axis,
    TensorGraph::TensorNode* src1,
    TensorGraph::TensorNode* src2,
    TensorGraph::TensorNode* dst)
{
    auto& src1_t = runtime.get_tensor<T>(src1);
    auto& src2_t = runtime.get_tensor<T>(src2);
    auto& dst_t = runtime.get_tensor<T>(dst);
    nntile::tensor::multiply_fiber<T>(alpha, src1_t, src2_t, dst_t, axis);
}

} // namespace

TensorGraph::TensorNode* multiply_fiber(
    Scalar alpha,
    TensorGraph::TensorNode* src1,
    TensorGraph::TensorNode* src2,
    const std::string& output_name,
    Index axis)
{
    if(src1 == nullptr || src2 == nullptr)
    {
        throw std::invalid_argument(
            "multiply_fiber: input tensors must be non-null");
    }
    if(src1->graph() != src2->graph())
    {
        throw std::invalid_argument(
            "multiply_fiber: input tensors must belong to the same graph");
    }
    if(src1->dtype() != src2->dtype())
    {
        throw std::invalid_argument(
            "multiply_fiber: input tensors must have the same dtype");
    }
    if(src1->ndim() != 1)
    {
        throw std::invalid_argument(
            "multiply_fiber: src1 must have ndim = 1");
    }
    if(axis < 0 || axis >= src2->ndim())
    {
        throw std::invalid_argument(
            "multiply_fiber: axis out of range");
    }

    validate_fiber_shape_and_merge(src1, src2, axis, 0, "multiply_fiber");

    std::vector<Index> output_shape = src2->shape();
    TensorGraph::TensorNode* dst = src1->graph()->data(
        std::move(output_shape),
        output_name,
        src1->dtype());
    dst->set_axes(src2->axes());

    auto op = std::make_shared<TensorMultiplyFiberOp>(
        alpha, src1, src2, dst, axis);
    src1->graph()->add_op(op);

    return dst;
}

void multiply_fiber(
    Scalar alpha,
    TensorGraph::TensorNode* src1,
    TensorGraph::TensorNode* src2,
    TensorGraph::TensorNode* dst,
    Index axis)
{
    if(src1 == nullptr || src2 == nullptr || dst == nullptr)
    {
        throw std::invalid_argument(
            "multiply_fiber: input tensors must be non-null");
    }
    if(src1->graph() != src2->graph() || src1->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "multiply_fiber: input tensors must belong to the same graph");
    }
    if(src1->dtype() != src2->dtype() || src1->dtype() != dst->dtype())
    {
        throw std::invalid_argument(
            "multiply_fiber: input tensors must have the same dtype");
    }
    if(src1->ndim() != 1)
    {
        throw std::invalid_argument(
            "multiply_fiber: src1 must have ndim = 1");
    }
    if(axis < 0 || axis >= src2->ndim())
    {
        throw std::invalid_argument(
            "multiply_fiber: axis out of range");
    }
    if(src1 == src2 || src1 == dst || src2 == dst)
    {
        throw std::invalid_argument(
            "multiply_fiber: src1, src2, and dst must be distinct tensors");
    }

    validate_fiber_shape_and_merge(src1, src2, axis, 0, "multiply_fiber");
    validate_same_shape_and_merge(src2, dst, "multiply_fiber");

    auto op = std::make_shared<TensorMultiplyFiberOp>(
        alpha, src1, src2, dst, axis);
    src1->graph()->add_op(op);
}

void TensorMultiplyFiberOp::execute(
    TensorGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(src1);

    switch(dtype)
    {
        case DataType::FP32:
            run_multiply_fiber<nntile::fp32_t>(
                runtime, alpha, axis, src1, src2, dst);
            break;
        case DataType::FP32_FAST_TF32:
            run_multiply_fiber<nntile::fp32_fast_tf32_t>(
                runtime, alpha, axis, src1, src2, dst);
            break;
        case DataType::FP32_FAST_FP16:
            run_multiply_fiber<nntile::fp32_fast_fp16_t>(
                runtime, alpha, axis, src1, src2, dst);
            break;
        case DataType::FP32_FAST_BF16:
            run_multiply_fiber<nntile::fp32_fast_bf16_t>(
                runtime, alpha, axis, src1, src2, dst);
            break;
        case DataType::FP64:
            run_multiply_fiber<nntile::fp64_t>(
                runtime, alpha, axis, src1, src2, dst);
            break;
        case DataType::FP16:
            run_multiply_fiber<nntile::fp16_t>(
                runtime, alpha, axis, src1, src2, dst);
            break;
        case DataType::BF16:
            run_multiply_fiber<nntile::bf16_t>(
                runtime, alpha, axis, src1, src2, dst);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for multiply_fiber operation");
        default:
            throw std::runtime_error("Unsupported data type for multiply_fiber");
    }
}

void TensorMultiplyFiberOp::lower_to_tile(const LoweringContext& ctx) const
{
    // Match nntile::tensor::multiply_fiber_async (src/tensor/multiply_fiber.cc).
    const TensorAxisLayout* lay_d = ctx.tiling.find(dst);
    if(lay_d == nullptr)
    {
        throw std::runtime_error(
            "lower_to_tile MULTIPLY_FIBER: missing tiling for dst");
    }

    tile_lower::assert_same_elementwise_layout(src2, dst, "MULTIPLY_FIBER src2/dst");

    const auto& tiles_s1 = tile_lower::tiles_of(ctx.tile_map, src1);
    const auto& tiles_s2 = tile_lower::tiles_of(ctx.tile_map, src2);
    const auto& tiles_d = tile_lower::tiles_of(ctx.tile_map, dst);

    std::vector<Index> dst_coord;

    for(Index lin_d = 0; lin_d < lay_d->grid_volume(); ++lin_d)
    {
        lay_d->grid_coord_from_linear(lin_d, dst_coord);
        const Index j = dst_coord[static_cast<size_t>(axis)];
        tile_graph::multiply_fiber(
            alpha,
            tiles_s1[static_cast<size_t>(j)],
            tiles_s2[static_cast<size_t>(lin_d)],
            tiles_d[static_cast<size_t>(lin_d)],
            axis);
    }
}

} // namespace nntile::graph::tensor
