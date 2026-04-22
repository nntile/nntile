/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/add_fiber_inplace.cc
 * TensorGraph add_fiber_inplace operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/add_fiber_inplace.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/add_fiber_inplace.hh"

#include "nntile/graph/tile/add_fiber_inplace.hh"
#include "nntile/graph/tile/lowering_context.hh"
#include "nntile/graph/tensor/tensor_graph_tiling.hh"
#include "nntile/graph/tensor/tile_lowering_helpers.hh"

namespace nntile::graph::tensor
{

namespace
{

template<typename T>
void run_add_fiber_inplace(
    TensorGraph::Runtime& runtime,
    Scalar alpha, Scalar beta,
    Index axis, Index batch_ndim,
    TensorGraph::TensorNode* fiber,
    TensorGraph::TensorNode* tensor)
{
    auto& fiber_t = runtime.get_tensor<T>(fiber);
    auto& tensor_t = runtime.get_tensor<T>(tensor);
    nntile::tensor::add_fiber_inplace<T>(
        alpha, fiber_t, beta, tensor_t, axis, batch_ndim);
}

} // namespace

void add_fiber_inplace(
    Scalar alpha,
    TensorGraph::TensorNode* fiber,
    Scalar beta,
    TensorGraph::TensorNode* tensor,
    Index axis,
    Index batch_ndim)
{
    if(fiber == nullptr || tensor == nullptr)
    {
        throw std::invalid_argument(
            "add_fiber_inplace: input tensors must be non-null");
    }
    if(fiber->graph() != tensor->graph())
    {
        throw std::invalid_argument(
            "add_fiber_inplace: input tensors must belong to the same graph");
    }
    if(fiber->dtype() != tensor->dtype())
    {
        throw std::invalid_argument(
            "add_fiber_inplace: input tensors must have the same dtype");
    }
    if(fiber == tensor)
    {
        throw std::invalid_argument(
            "add_fiber_inplace: fiber and tensor must be distinct tensors");
    }
    validate_fiber_shape_and_merge(fiber, tensor, axis, batch_ndim,
                                   "add_fiber_inplace");

    auto op = std::make_shared<TensorAddFiberInplaceOp>(
        fiber, tensor, alpha, beta, axis, batch_ndim);
    tensor->graph()->add_op(op);
}

void TensorAddFiberInplaceOp::execute(
    TensorGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(fiber);

    switch(dtype)
    {
        case DataType::FP32:
            run_add_fiber_inplace<nntile::fp32_t>(
                runtime, alpha, beta, axis, batch_ndim, fiber, tensor);
            break;
        case DataType::FP32_FAST_TF32:
            run_add_fiber_inplace<nntile::fp32_fast_tf32_t>(
                runtime, alpha, beta, axis, batch_ndim, fiber, tensor);
            break;
        case DataType::FP32_FAST_FP16:
            run_add_fiber_inplace<nntile::fp32_fast_fp16_t>(
                runtime, alpha, beta, axis, batch_ndim, fiber, tensor);
            break;
        case DataType::FP32_FAST_BF16:
            run_add_fiber_inplace<nntile::fp32_fast_bf16_t>(
                runtime, alpha, beta, axis, batch_ndim, fiber, tensor);
            break;
        case DataType::FP64:
            run_add_fiber_inplace<nntile::fp64_t>(
                runtime, alpha, beta, axis, batch_ndim, fiber, tensor);
            break;
        case DataType::FP16:
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for add_fiber_inplace operation");
        case DataType::BF16:
            run_add_fiber_inplace<nntile::bf16_t>(
                runtime, alpha, beta, axis, batch_ndim, fiber, tensor);
            break;
        default:
            throw std::runtime_error(
                "Unsupported data type for add_fiber_inplace");
    }
}

void TensorAddFiberInplaceOp::lower_to_tile(const LoweringContext& ctx) const
{
    // Match nntile::tensor::add_fiber_inplace_async (src/tensor/add_fiber_inplace.cc).
    if(alpha == 0.0)
    {
        return;
    }

    const TensorAxisLayout* lay_d = ctx.tiling.find(tensor);
    const TensorAxisLayout* lay_f = ctx.tiling.find(fiber);
    if(lay_d == nullptr || lay_f == nullptr)
    {
        throw std::runtime_error(
            "lower_to_tile ADD_FIBER_INPLACE: missing tiling for tensor and/or "
            "fiber");
    }

    const auto& tiles_f = tile_lower::tiles_of(ctx.tile_map, fiber);
    const auto& tiles_t = tile_lower::tiles_of(ctx.tile_map, tensor);

    std::vector<Index> dst_coord;
    std::vector<Index> fiber_coord(static_cast<size_t>(fiber->ndim()));

    for(Index lin_d = 0; lin_d < lay_d->grid_volume(); ++lin_d)
    {
        lay_d->grid_coord_from_linear(lin_d, dst_coord);
        const Index j = dst_coord[static_cast<size_t>(axis)];
        fiber_coord[0] = j;
        for(Index b = 0; b < batch_ndim; ++b)
        {
            fiber_coord[static_cast<size_t>(b + 1)] =
                dst_coord[static_cast<size_t>(tensor->ndim() - batch_ndim + b)];
        }
        const Index lin_f = lay_f->grid_linear(fiber_coord);
        tile_graph::add_fiber_inplace(
            alpha,
            tiles_f[static_cast<size_t>(lin_f)],
            beta,
            tiles_t[static_cast<size_t>(lin_d)],
            axis,
            batch_ndim);
    }
}

} // namespace nntile::graph::tensor
