/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file nntile/src/tensor_graph/sqrt.cc
 * TensorGraph Sqrt operation implementation.
 *
 * @version 1.1.0
 * */
/*!
 * INTENTIONALLY DISABLED in the NNTile 2.0 transformer-only scope.
 * Kept in-tree for future re-enablement. No product path calls this code;
 * its tests are not registered while the family is in
 * NNTILE_DISABLED_KERNELS.
 */
#include <nntile/common.hh>
#include "nntile/kernel_scope.hh"
#include "nntile/tensor/ops/sqrt.hh"

#include "nntile/base_types.hh"
#include "nntile/dtype.hh"
#include "nntile/tensor.hh"
#include "nntile/tensor/ops/sqrt.hh"

#include <nntile/tensor/tile_lowering_helpers.hh>
#include <nntile/tile/graph_ops.hh>
#include <nntile/kernel_scope.hh>
#include <stdexcept>
#include <utility>

namespace nntile::tensor
{

TensorGraph::TensorNode *sqrt(TensorGraph::TensorNode *src)
{
    nntile::require_kernel_family("sqrt");
    if (src == nullptr)
    {
        throw std::invalid_argument("sqrt: input tensor must be non-null");
    }

    std::vector<Index> output_shape = src->shape();
    TensorGraph::TensorNode *output =
        src->graph()->emplace_data(std::move(output_shape), src->dtype());
    output->set_axes(src->axes());

    auto op = std::make_shared<TensorSqrtOp>(src, output);
    src->graph()->add_op(op);

    return output;
}

void sqrt(TensorGraph::TensorNode *src, TensorGraph::TensorNode *dst)
{
    nntile::require_kernel_family("sqrt");
    if (src == nullptr || dst == nullptr)
    {
        throw std::invalid_argument("sqrt: input tensors must be non-null");
    }
    if (src->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "sqrt: input tensors must belong to the same graph");
    }
    if (src->dtype() != dst->dtype())
    {
        throw std::invalid_argument(
            "sqrt: input tensors must have the same dtype");
    }
    if (src == dst)
    {
        throw std::invalid_argument(
            "sqrt: src and dst must be distinct tensors");
    }
    validate_same_shape_and_merge(src, dst, "sqrt");

    auto op = std::make_shared<TensorSqrtOp>(src, dst);
    src->graph()->add_op(op);
}

void TensorSqrtOp::lower_to_tile(const LoweringContext &ctx) const
{
    tile_lower::lower_unary2(src, dst, ctx.tile_map, "SQRT", tile::sqrt);
}

} // namespace nntile::tensor
