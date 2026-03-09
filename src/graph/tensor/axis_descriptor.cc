/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/axis_descriptor.cc
 * AxisDescriptor and merge_axis implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/axis_descriptor.hh"
#include "nntile/graph/tensor/graph.hh"

#include <stdexcept>

namespace nntile::graph
{

void merge_axis(std::shared_ptr<AxisDescriptor>& keep,
                std::shared_ptr<AxisDescriptor>& replace)
{
    if(keep == replace)
    {
        return;
    }
    if(keep->extent != replace->extent)
    {
        throw std::invalid_argument(
            "merge_axis: cannot merge axes with different extents (" +
            std::to_string(keep->extent) + " vs " +
            std::to_string(replace->extent) + ")");
    }

    if(keep->name.empty() && !replace->name.empty())
    {
        keep->name = replace->name;
    }

    // Save the descriptor being replaced — the `replace` reference may
    // alias one of the tensor slots we reassign inside the loop.
    std::shared_ptr<AxisDescriptor> old_desc = replace;

    for(auto [node_ptr, axis_idx] : old_desc->members)
    {
        auto* node = static_cast<TensorGraph::TensorNode*>(node_ptr);
        node->mutable_axes()[static_cast<size_t>(axis_idx)] = keep;
        keep->members.push_back({node_ptr, axis_idx});
    }
    old_desc->members.clear();
}

} // namespace nntile::graph
