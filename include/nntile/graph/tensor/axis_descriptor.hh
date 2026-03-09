/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/axis_descriptor.hh
 * AxisDescriptor - shared descriptor for a dimension group in TensorGraph.
 *
 * @version 1.1.0
 * */

#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nntile/base_types.hh>

namespace nntile::graph
{

class TensorGraph;

//! Shared descriptor for a group of tensor dimensions that must be
//! tiled identically. Tensors sharing the same AxisDescriptor (via
//! shared_ptr) belong to the same dimension group.
struct AxisDescriptor
{
    Index extent;
    std::string name;

    //! (tensor_node, axis_index) pairs for all members of this group.
    //! Updated during merge_axis().
    std::vector<std::pair<void*, int>> members;
};

//! Merge two axis groups. All tensors holding `replace` are redirected
//! to hold `keep`. Throws if extents differ. No-op if already same.
void merge_axis(std::shared_ptr<AxisDescriptor>& keep,
                std::shared_ptr<AxisDescriptor>& replace);

} // namespace nntile::graph
