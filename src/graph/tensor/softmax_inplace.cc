/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/softmax_inplace.cc
 * TensorGraph softmax_inplace operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/softmax_inplace.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/graph/tensor/tensor_graph_tiling.hh"
#include "nntile/graph/tensor/tile_lowering_helpers.hh"
#include "nntile/graph/tile/softmax_inplace.hh"
#include "nntile/tensor/softmax_inplace.hh"

namespace nntile::graph::tensor
{

namespace
{

template<typename T>
void run_softmax_inplace(
    TensorGraph::Runtime& runtime,
    Scalar alpha, Index axis,
    TensorGraph::TensorNode* maxsumexp,
    TensorGraph::TensorNode* dst)
{
    auto& maxsumexp_t = runtime.get_tensor<T>(maxsumexp);
    auto& dst_t = runtime.get_tensor<T>(dst);
    nntile::tensor::softmax_inplace<T>(maxsumexp_t, alpha, dst_t, axis);
}

} // namespace

void softmax_inplace(
    TensorGraph::TensorNode* maxsumexp,
    TensorGraph::TensorNode* dst,
    Scalar alpha,
    Index axis)
{
    if(maxsumexp == nullptr || dst == nullptr)
    {
        throw std::invalid_argument(
            "softmax_inplace: input tensors must be non-null");
    }
    if(maxsumexp->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "softmax_inplace: input tensors must belong to the same graph");
    }
    if(maxsumexp->dtype() != dst->dtype())
    {
        throw std::invalid_argument(
            "softmax_inplace: input tensors must have the same dtype");
    }
    // maxsumexp has shape with 2 at axis, dst has full shape

    auto op = std::make_shared<TensorSoftmaxInplaceOp>(
        maxsumexp, dst, alpha, axis);
    maxsumexp->graph()->add_op(op);
}

void TensorSoftmaxInplaceOp::execute(
    TensorGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(maxsumexp);

    switch(dtype)
    {
        case DataType::FP32:
            run_softmax_inplace<nntile::fp32_t>(
                runtime, alpha, axis, maxsumexp, dst);
            break;
        case DataType::FP32_FAST_TF32:
            run_softmax_inplace<nntile::fp32_fast_tf32_t>(
                runtime, alpha, axis, maxsumexp, dst);
            break;
        case DataType::FP32_FAST_FP16:
            run_softmax_inplace<nntile::fp32_fast_fp16_t>(
                runtime, alpha, axis, maxsumexp, dst);
            break;
        case DataType::FP32_FAST_BF16:
            run_softmax_inplace<nntile::fp32_fast_bf16_t>(
                runtime, alpha, axis, maxsumexp, dst);
            break;
        case DataType::FP64:
            run_softmax_inplace<nntile::fp64_t>(
                runtime, alpha, axis, maxsumexp, dst);
            break;
        case DataType::FP16:
            run_softmax_inplace<nntile::fp16_t>(
                runtime, alpha, axis, maxsumexp, dst);
            break;
        case DataType::BF16:
            run_softmax_inplace<nntile::bf16_t>(
                runtime, alpha, axis, maxsumexp, dst);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for softmax_inplace operation");
        default:
            throw std::runtime_error(
                "Unsupported data type for softmax_inplace");
    }
}

void TensorSoftmaxInplaceOp::lower_to_tile(const LoweringContext& ctx) const
{
    // Match nntile::tensor::softmax_inplace_async: iterate maxsumexp tiles,
    // then all dst tiles along `axis` sharing that fiber (see
    // src/tensor/softmax_inplace.cc and src/tensor/softmax.cc).
    const TensorAxisLayout* lay_m = ctx.tiling.find(maxsumexp);
    const TensorAxisLayout* lay_d = ctx.tiling.find(dst);
    if(lay_m == nullptr || lay_d == nullptr)
    {
        throw std::runtime_error(
            "lower_to_tile SOFTMAX_INPLACE: missing tiling for maxsumexp "
            "and/or dst");
    }

    const auto& tiles_m = tile_lower::tiles_of(ctx.tile_map, maxsumexp);
    const auto& tiles_d = tile_lower::tiles_of(ctx.tile_map, dst);

    std::vector<Index> m_coord;
    std::vector<Index> dst_coord(static_cast<size_t>(dst->ndim()));

    for(Index lin_m = 0; lin_m < lay_m->grid_volume(); ++lin_m)
    {
        lay_m->grid_coord_from_linear(lin_m, m_coord);
        TileGraph::TileNode* m_tile = tiles_m[static_cast<size_t>(lin_m)];

        for(Index j = 0; j < axis; ++j)
        {
            dst_coord[static_cast<size_t>(j)] =
                m_coord[static_cast<size_t>(j + 1)];
        }
        for(Index j = axis + 1; j < dst->ndim(); ++j)
        {
            dst_coord[static_cast<size_t>(j)] =
                m_coord[static_cast<size_t>(j)];
        }

        const Index nseg_along_axis =
            lay_d->grid_shape()[static_cast<size_t>(axis)];
        for(Index j = 0; j < nseg_along_axis; ++j)
        {
            dst_coord[static_cast<size_t>(axis)] = j;
            const Index lin_d = lay_d->grid_linear(dst_coord);
            TileGraph::TileNode* d_tile =
                tiles_d[static_cast<size_t>(lin_d)];
            tile_graph::softmax_inplace(m_tile, alpha, d_tile, axis);
        }
    }
}

} // namespace nntile::graph::tensor
