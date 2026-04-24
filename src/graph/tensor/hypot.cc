/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/hypot.cc
 * TensorGraph hypot operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/hypot.hh"

#include <stdexcept>
#include <utility>

#include "nntile/base_types.hh"
#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/graph/tensor/tile_lowering_helpers.hh"
#include "nntile/graph/tile/hypot.hh"
#include "nntile/graph/tile/lowering_context.hh"
#include "nntile/tensor/hypot.hh"

namespace nntile::graph::tensor
{

void TensorHypotOp::lower_to_tile(const LoweringContext& ctx) const
{
    const auto& m = ctx.tile_map;
    const auto& v1 = tile_lower::tiles_of(m, src1);
    const auto& v2 = tile_lower::tiles_of(m, src2);
    const auto& vd = tile_lower::tiles_of(m, dst);
    if(v1.size() != v2.size() || v1.size() != vd.size())
    {
        throw std::runtime_error("lower_to_tile HYPOT: tile count mismatch");
    }
    tile_lower::assert_same_elementwise_layout(src1, src2, "HYPOT src1/src2");
    tile_lower::assert_same_elementwise_layout(src1, dst, "HYPOT src1/dst");
    for(size_t i = 0; i < v1.size(); ++i)
    {
        tile_graph::hypot(alpha, v1[i], beta, v2[i], vd[i]);
    }
}

TensorGraph::TensorNode* hypot(
    Scalar alpha,
    TensorGraph::TensorNode* src1,
    Scalar beta,
    TensorGraph::TensorNode* src2,
    const std::string& output_name)
{
    if(src1 == nullptr || src2 == nullptr)
    {
        throw std::invalid_argument(
            "hypot: input tensors must be non-null");
    }
    if(src1 == src2)
    {
        throw std::invalid_argument(
            "hypot: src1 and src2 must be distinct tensors");
    }
    if(src1->graph() != src2->graph())
    {
        throw std::invalid_argument(
            "hypot: input tensors must belong to the same graph");
    }
    if(src1->dtype() != src2->dtype())
    {
        throw std::invalid_argument(
            "hypot: input tensors must have the same dtype");
    }
    validate_same_shape_and_merge(src1, src2, "hypot");

    std::vector<Index> output_shape = src1->shape();
    TensorGraph::TensorNode* dst = src1->graph()->data(
        std::move(output_shape),
        output_name,
        src1->dtype());
    dst->set_axes(src1->axes());

    hypot(alpha, src1, beta, src2, dst);

    return dst;
}

void hypot(
    Scalar alpha,
    TensorGraph::TensorNode* src1,
    Scalar beta,
    TensorGraph::TensorNode* src2,
    TensorGraph::TensorNode* dst)
{
    if(src1 == nullptr || src2 == nullptr || dst == nullptr)
    {
        throw std::invalid_argument(
            "hypot: input tensors must be non-null");
    }
    if(src1->graph() != src2->graph() || src1->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "hypot: input tensors must belong to the same graph");
    }
    if(src1->dtype() != src2->dtype() || src1->dtype() != dst->dtype())
    {
        throw std::invalid_argument(
            "hypot: input tensors must have the same dtype");
    }
    if(src1 == src2 || src1 == dst || src2 == dst)
    {
        throw std::invalid_argument(
            "hypot: src1, src2, and dst must be distinct tensors");
    }
    validate_same_shape_and_merge(src1, src2, "hypot");
    validate_same_shape_and_merge(src1, dst, "hypot");

    auto op = std::make_shared<TensorHypotOp>(
        alpha, beta, src1, src2, dst);
    src1->graph()->add_op(op);
}

} // namespace nntile::graph::tensor
