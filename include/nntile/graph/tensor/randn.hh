/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/randn.hh
 * TensorGraph randn operation: (dst, start, underlying_shape, seed, mean, stddev)
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <vector>

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/tensor/graph.hh>

namespace nntile::graph
{
struct LoweringContext;
}

namespace nntile::graph::tensor
{

//! Randn operation: fill dst with random normal values
struct TensorRandnOp : TensorGraph::OpNode
{
    std::vector<Index> start;
    std::vector<Index> underlying_shape;
    unsigned long long seed;
    Scalar mean;
    Scalar stddev;
    TensorGraph::TensorNode* dst = nullptr;

    TensorRandnOp() = default;
    TensorRandnOp(
        TensorGraph::TensorNode* dst_,
        const std::vector<Index>& start_,
        const std::vector<Index>& underlying_shape_,
        unsigned long long seed_,
        Scalar mean_,
        Scalar stddev_)
        : start(start_), underlying_shape(underlying_shape_)
        , seed(seed_), mean(mean_), stddev(stddev_)
        , dst(dst_)
    {
        inputs_ = {dst};
        outputs_ = {dst};
    }

    std::string op_name() const override { return "RANDN"; }


    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorRandnOp>(*this);
    }

    void lower_to_tile(const LoweringContext& ctx) const override;
};

void randn(
    TensorGraph::TensorNode* dst,
    const std::vector<Index>& start,
    const std::vector<Index>& underlying_shape,
    unsigned long long seed,
    Scalar mean,
    Scalar stddev);

} // namespace nntile::graph::tensor
