/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/copy_intersection.hh
 * TensorGraph copy_intersection: copy overlapping region src->dst
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <vector>

// NNTile headers
#include <nntile/graph/tensor/graph.hh>

namespace nntile::graph
{
struct LoweringContext;
}

namespace nntile::graph::tensor
{

//! Copy intersection: copy overlapping region from src to dst
struct TensorCopyIntersectionOp : TensorGraph::OpNode
{
    TensorGraph::TensorNode* src = nullptr;
    std::vector<Index> src_offset;
    TensorGraph::TensorNode* dst = nullptr;
    std::vector<Index> dst_offset;

    TensorCopyIntersectionOp() = default;
    TensorCopyIntersectionOp(TensorGraph::TensorNode* src_,
                            std::vector<Index> src_offset_,
                            TensorGraph::TensorNode* dst_,
                            std::vector<Index> dst_offset_)
        : src(src_), src_offset(std::move(src_offset_)),
          dst(dst_), dst_offset(std::move(dst_offset_))
    {
        inputs_ = {src, dst};
        outputs_ = {dst};
    }

    std::string op_name() const override { return "COPY_INTERSECTION"; }

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorCopyIntersectionOp>(*this);
    }

    void lower_to_tile(const LoweringContext& ctx) const override;
};

//! Copy intersection: copy overlapping region from src to dst
void copy_intersection(TensorGraph::TensorNode* src,
                       const std::vector<Index>& src_offset,
                       TensorGraph::TensorNode* dst,
                       const std::vector<Index>& dst_offset);

} // namespace nntile::graph::tensor
