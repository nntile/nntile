/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/copy.cc
 * TensorGraph copy operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/copy.hh"

#include <stdexcept>
#include <utility>

#include "nntile/graph/tensor.hh"
#include "nntile/graph/tensor/tile_lowering_helpers.hh"
#include "nntile/graph/tile/copy.hh"
#include "nntile/tensor/copy.hh"

namespace nntile::graph::tensor
{



TensorGraph::TensorNode* copy(TensorGraph::TensorNode* src,
                              const std::string& output_name)
{
    if(src == nullptr)
        throw std::invalid_argument("copy: input tensor must be non-null");
    std::vector<Index> output_shape = src->shape();
    TensorGraph::TensorNode* output = src->graph()->data(
        std::move(output_shape), output_name, src->dtype());
    output->set_axes(src->axes());
    copy(src, output);
    return output;
}

void copy(TensorGraph::TensorNode* src, TensorGraph::TensorNode* dst)
{
    if(src == nullptr || dst == nullptr)
        throw std::invalid_argument("copy: tensors must be non-null");
    if(src->graph() != dst->graph())
        throw std::invalid_argument("copy: tensors must belong to same graph");
    if(src->dtype() != dst->dtype())
        throw std::invalid_argument("copy: tensors must have same dtype");
    if(src == dst)
        throw std::invalid_argument("copy: src and dst must be distinct tensors");
    validate_same_shape_and_merge(src, dst, "copy");

    auto op = std::make_shared<TensorCopyOp>(src, dst);
    src->graph()->add_op(op);
}

void TensorCopyOp::lower_to_tile(const LoweringContext& ctx) const
{
    const auto& m = ctx.tile_map;
    const auto& v_src = tile_lower::tiles_of(m, src);
    const auto& v_dst = tile_lower::tiles_of(m, dst);
    if(v_src.size() != v_dst.size())
    {
        throw std::runtime_error(
            "lower_to_tile COPY: tile count mismatch for src/dst");
    }
    tile_lower::assert_same_elementwise_layout(src, dst, "COPY src/dst");
    for(size_t i = 0; i < v_src.size(); ++i)
    {
        tile_graph::copy(v_src[i], v_dst[i]);
    }
}

} // namespace nntile::graph::tensor
