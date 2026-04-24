/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/randn.hh
 * TileGraph randn operation: (dst, start, underlying_shape, seed, mean, stddev)
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

//! Randn operation: fill dst with random normal values
struct TileRandnOp : TileGraph::OpNode
{
    std::vector<Index> start;
    std::vector<Index> underlying_shape;
    unsigned long long seed = 0;
    Scalar mean = 0.0;
    Scalar stddev = 0.0;
    TileGraph::TileNode* dst = nullptr;
    TileRandnOp() = default;
    TileRandnOp(const std::vector<Index>& st,
        const std::vector<Index>& us,
        unsigned long long sd,
        Scalar m,
        Scalar s,
        TileGraph::TileNode* d) : start(st), underlying_shape(us), seed(sd), mean(m), stddev(s), dst(d)
    {
        inputs_ = {};
        outputs_ = {dst};
    }
    std::string op_name() const override { return "TILE_RANDN"; }
    void execute(TileGraph::Runtime& runtime) const override;
    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileRandnOp>(*this);
    }
};
void randn(TileGraph::TileNode* dst, const std::vector<Index>& start, const std::vector<Index>& underlying_shape,
    unsigned long long seed, Scalar mean, Scalar stddev);
} // namespace nntile::graph::tile_graph
