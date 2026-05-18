/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/scale.cc
 * TensorGraph scale operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/scale.hh"

#include <stdexcept>
#include <utility>

#include "nntile/base_types.hh"
#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/scale.hh"

#include "nntile/graph/tile/lowering_context.hh"
#include "nntile/graph/tensor/tensor_graph_tiling.hh"
#include "nntile/graph/tensor/tile_lowering_helpers.hh"
#include "nntile/graph/tile/scale.hh"

namespace nntile::graph::tensor
{



TensorGraph::TensorNode* scale(Scalar alpha, TensorGraph::TensorNode* src,
                               const std::string& output_name)
{
    if(src == nullptr)
        throw std::invalid_argument("scale: input tensor must be non-null");
    std::vector<Index> output_shape = src->shape();
    TensorGraph::TensorNode* output = src->graph()->data(
        std::move(output_shape), output_name, src->dtype());
    output->set_axes(src->axes());
    scale(alpha, src, output);
    return output;
}

void scale(Scalar alpha, TensorGraph::TensorNode* src,
           TensorGraph::TensorNode* dst)
{
    if(src == nullptr || dst == nullptr)
        throw std::invalid_argument("scale: tensors must be non-null");
    if(src->graph() != dst->graph())
        throw std::invalid_argument("scale: tensors must belong to same graph");
    if(src->dtype() != dst->dtype())
        throw std::invalid_argument("scale: tensors must have same dtype");
    if(src == dst)
        throw std::invalid_argument("scale: src and dst must be distinct tensors");
    validate_same_shape_and_merge(src, dst, "scale");

    auto op = std::make_shared<TensorScaleOp>(src, dst, alpha);
    src->graph()->add_op(op);
}

void TensorScaleOp::lower_to_tile(const LoweringContext& ctx) const
{
    const auto& tiles_src = tile_lower::tiles_of(ctx.tile_map, src);
    const auto& tiles_dst = tile_lower::tiles_of(ctx.tile_map, dst);
    if(tiles_src.size() != tiles_dst.size())
    {
        throw std::runtime_error("lower_to_tile SCALE: tile count mismatch");
    }
    tile_lower::assert_same_elementwise_layout(src, dst, "SCALE src/dst");
    for(size_t i = 0; i < tiles_src.size(); ++i)
    {
        tile_graph::scale(alpha, tiles_src[i], tiles_dst[i]);
    }
}

} // namespace nntile::graph::tensor
