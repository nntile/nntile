/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/copy_intersection.hh
 * TileGraph copy_intersection: copy overlapping region src->dst
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <vector>

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/tile/graph.hh>

namespace nntile::graph::tile_graph
{

//! Copy intersection of two tiles (with int64 scratch)
struct TileCopyIntersectionOp : TileGraph::OpNode
{
    std::vector<Index> src_offset;
    std::vector<Index> dst_offset;
    TileGraph::TileNode* src = nullptr;
    TileGraph::TileNode* dst = nullptr;
    TileGraph::TileNode* scratch = nullptr;

    TileCopyIntersectionOp() = default;
    TileCopyIntersectionOp(
        const std::vector<Index>& so,
        const std::vector<Index>& dof,
        TileGraph::TileNode* s,
        TileGraph::TileNode* d,
        TileGraph::TileNode* sc)
        : src_offset(so)
        , dst_offset(dof)
        , src(s)
        , dst(d)
        , scratch(sc)
    {
        inputs_ = {src, dst, scratch};
        outputs_ = {dst};
    }

    std::string op_name() const override { return "TILE_COPY_INTERSECTION"; }
    void execute(TileGraph::Runtime& runtime) const override;
    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileCopyIntersectionOp>(*this);
    }
};

//! Copy intersection: copy overlapping region from src to dst
void copy_intersection(
    TileGraph::TileNode* src,
    const std::vector<Index>& src_offset,
    TileGraph::TileNode* dst,
    const std::vector<Index>& dst_offset,
    TileGraph::TileNode* scratch);
} // namespace nntile::graph::tile_graph
