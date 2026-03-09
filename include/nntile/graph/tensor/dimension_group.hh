/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/dimension_group.hh
 * Automatic discovery of dimension groups (equivalence classes) across
 * a TensorGraph, based on operation-level constraints.
 *
 * @version 1.1.0
 * */

#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

#include <nntile/base_types.hh>
#include <nntile/graph/tensor/graph_decl.hh>

namespace nntile::graph
{

//! A reference to a specific dimension of a specific tensor.
struct DimId
{
    const TensorGraph::TensorNode* node;
    int axis;

    bool operator==(const DimId& other) const
    {
        return node == other.node && axis == other.axis;
    }
    bool operator<(const DimId& other) const
    {
        if(node != other.node)
            return node < other.node;
        return axis < other.axis;
    }
};

//! Constraint: two dimensions must belong to the same tiling group.
struct DimConstraint
{
    DimId a;
    DimId b;
};

//! A group of tensor dimensions that must be tiled identically.
struct DimensionGroup
{
    std::vector<DimId> members;
    Index extent;
    std::string name;
};

//! Extract tiling constraints from a single operation.
//! Returns an empty vector for ops with no discoverable constraints.
std::vector<DimConstraint> get_tiling_constraints(
    const TensorGraph::OpNode& op);

//! Discovers dimension groups for the entire graph by walking all
//! operations and merging constrained dimensions via union-find.
//!
//! Each group contains all (tensor_node, axis) pairs that must share
//! the same tile size. Groups can be named afterwards by looking up
//! a specific (tensor_name, axis) pair.
std::vector<DimensionGroup> discover_dimension_groups(
    const TensorGraph& graph);

} // namespace nntile::graph
