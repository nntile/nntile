/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/norm_fiber.cc
 * TensorGraph norm_fiber operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/norm_fiber.hh"

#include <stdexcept>
#include <utility>

#include "nntile/base_types.hh"
#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/norm_fiber.hh"

#include "nntile/graph/tile/lowering_context.hh"
#include "nntile/graph/tile/norm_fiber.hh"
#include "nntile/graph/tile/norm_fiber_inplace.hh"
#include "nntile/graph/tensor/tensor_graph_tiling.hh"
#include "nntile/graph/tensor/tile_lowering_helpers.hh"

namespace nntile::graph::tensor
{

namespace
{

template<typename T>
void run_norm_fiber(
    TensorGraph::Runtime& runtime,
    Scalar alpha, Scalar beta,
    Index axis, Index batch_ndim, int redux,
    TensorGraph::TensorNode* src1,
    TensorGraph::TensorNode* src2,
    TensorGraph::TensorNode* dst)
{
    auto& src1_t = runtime.get_tensor<T>(src1);
    auto& src2_t = runtime.get_tensor<T>(src2);
    auto& dst_t = runtime.get_tensor<T>(dst);
    nntile::tensor::norm_fiber<T>(
        alpha, src1_t, beta, src2_t, dst_t, axis, batch_ndim, redux);
}

} // namespace

TensorGraph::TensorNode* norm_fiber(
    Scalar alpha,
    TensorGraph::TensorNode* src1,
    Scalar beta,
    TensorGraph::TensorNode* src2,
    const std::string& output_name,
    Index axis,
    Index batch_ndim,
    int redux)
{
    if(src1 == nullptr || src2 == nullptr)
    {
        throw std::invalid_argument(
            "norm_fiber: input tensors must be non-null");
    }
    if(src1->graph() != src2->graph())
    {
        throw std::invalid_argument(
            "norm_fiber: input tensors must belong to the same graph");
    }
    if(src1->dtype() != src2->dtype())
    {
        throw std::invalid_argument(
            "norm_fiber: input tensors must have the same dtype");
    }

    std::vector<Index> output_shape = src2->shape();
    TensorGraph::TensorNode* dst = src1->graph()->data(
        std::move(output_shape),
        output_name,
        src1->dtype());

    validate_fiber_shape_and_merge(dst, src1, axis, batch_ndim, "norm_fiber");
    validate_same_shape_and_merge(src2, dst, "norm_fiber");

    auto op = std::make_shared<TensorNormFiberOp>(
        alpha, beta, src1, src2, dst, axis, batch_ndim, redux);
    src1->graph()->add_op(op);

    return dst;
}

void norm_fiber(
    Scalar alpha,
    TensorGraph::TensorNode* src1,
    Scalar beta,
    TensorGraph::TensorNode* src2,
    TensorGraph::TensorNode* dst,
    Index axis,
    Index batch_ndim,
    int redux)
{
    if(src1 == nullptr || src2 == nullptr || dst == nullptr)
    {
        throw std::invalid_argument(
            "norm_fiber: input tensors must be non-null");
    }
    if(src1->graph() != src2->graph() || src1->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "norm_fiber: input tensors must belong to the same graph");
    }
    if(src1->dtype() != src2->dtype() || src1->dtype() != dst->dtype())
    {
        throw std::invalid_argument(
            "norm_fiber: input tensors must have the same dtype");
    }
    if(src1 == src2 || src1 == dst || src2 == dst)
    {
        throw std::invalid_argument(
            "norm_fiber: src1, src2, and dst must be distinct tensors");
    }

    validate_fiber_shape_and_merge(dst, src1, axis, batch_ndim, "norm_fiber");
    validate_same_shape_and_merge(src2, dst, "norm_fiber");

    auto op = std::make_shared<TensorNormFiberOp>(
        alpha, beta, src1, src2, dst, axis, batch_ndim, redux);
    src1->graph()->add_op(op);
}

void TensorNormFiberOp::execute(
    TensorGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(src1);

    switch(dtype)
    {
        case DataType::FP32:
            run_norm_fiber<nntile::fp32_t>(
                runtime, alpha, beta, axis, batch_ndim, redux, src1, src2, dst);
            break;
        case DataType::FP32_FAST_TF32:
            run_norm_fiber<nntile::fp32_fast_tf32_t>(
                runtime, alpha, beta, axis, batch_ndim, redux, src1, src2, dst);
            break;
        case DataType::FP32_FAST_FP16:
            run_norm_fiber<nntile::fp32_fast_fp16_t>(
                runtime, alpha, beta, axis, batch_ndim, redux, src1, src2, dst);
            break;
        case DataType::FP32_FAST_BF16:
            run_norm_fiber<nntile::fp32_fast_bf16_t>(
                runtime, alpha, beta, axis, batch_ndim, redux, src1, src2, dst);
            break;
        case DataType::FP64:
            run_norm_fiber<nntile::fp64_t>(
                runtime, alpha, beta, axis, batch_ndim, redux, src1, src2, dst);
            break;
        case DataType::FP16:
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for norm_fiber operation");
        case DataType::BF16:
            run_norm_fiber<nntile::bf16_t>(
                runtime, alpha, beta, axis, batch_ndim, redux, src1, src2, dst);
            break;
        default:
            throw std::runtime_error("Unsupported data type for norm_fiber");
    }
}

void TensorNormFiberOp::lower_to_tile(const LoweringContext& ctx) const
{
    // Match nntile::tensor::norm_fiber_async (src/tensor/norm_fiber.cc).
    const TensorAxisLayout* lay1 = ctx.tiling.find(src1);
    if(lay1 == nullptr)
    {
        throw std::runtime_error("lower_to_tile NORM_FIBER: missing tiling for src1");
    }

    tile_lower::assert_same_elementwise_layout(src2, dst, "NORM_FIBER src2/dst");

    const TensorAxisLayout* lay_d = ctx.tiling.find(dst);
    if(lay_d == nullptr)
    {
        throw std::runtime_error("lower_to_tile NORM_FIBER: missing tiling for dst");
    }

    const auto& tiles_s1 = tile_lower::tiles_of(ctx.tile_map, src1);
    const auto& tiles_s2 = tile_lower::tiles_of(ctx.tile_map, src2);
    const auto& tiles_d = tile_lower::tiles_of(ctx.tile_map, dst);

    constexpr Scalar one = 1.0;
    std::vector<Index> s1_coord;
    std::vector<Index> dst_coord(static_cast<size_t>(dst->ndim()));

    for(Index lin1 = 0; lin1 < lay1->grid_volume(); ++lin1)
    {
        lay1->grid_coord_from_linear(lin1, s1_coord);
        bool init_first = true;
        for(Index j = 0; j < src1->ndim() - batch_ndim; ++j)
        {
            if(j != axis && s1_coord[static_cast<size_t>(j)] != 0)
            {
                init_first = false;
                break;
            }
        }

        dst_coord[0] = s1_coord[static_cast<size_t>(axis)];
        for(Index b = 0; b < batch_ndim; ++b)
        {
            dst_coord[static_cast<size_t>(b + 1)] =
                s1_coord[static_cast<size_t>(src1->ndim() - batch_ndim + b)];
        }
        const Index lin_d = lay_d->grid_linear(dst_coord);

        if(init_first)
        {
            tile_graph::norm_fiber(
                alpha,
                tiles_s1[static_cast<size_t>(lin1)],
                beta,
                tiles_s2[static_cast<size_t>(lin_d)],
                tiles_d[static_cast<size_t>(lin_d)],
                axis,
                batch_ndim,
                redux);
        }
        else
        {
            tile_graph::norm_fiber_inplace(
                alpha,
                tiles_s1[static_cast<size_t>(lin1)],
                one,
                tiles_d[static_cast<size_t>(lin_d)],
                axis,
                batch_ndim,
                redux);
        }
    }
}

} // namespace nntile::graph::tensor
