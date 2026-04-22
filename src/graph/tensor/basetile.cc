/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/basetile.cc
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/basetile.hh"

#include <numeric>
#include <stdexcept>

#include "nntile/graph/tensor/axis_descriptor.hh"
#include "nntile/graph/tensor/graph_data_node.hh"

namespace nntile::graph
{

std::vector<Index> compute_basetile_shape_for_tensor(
    const TensorGraph::TensorNode* node)
{
    const std::vector<Index>& shape = node->shape();
    const auto& axes = node->axes();
    if(axes.size() != shape.size())
    {
        throw std::runtime_error(
            "compute_basetile_shape_for_tensor: axes size mismatch for '" +
            node->name() + "'");
    }
    std::vector<Index> basetile(shape.size());
    for(size_t i = 0; i < shape.size(); ++i)
    {
        const AxisDescriptor* ax = axes[i].get();
        if(!ax->is_tiled())
        {
            basetile[i] = shape[i];
            continue;
        }
        const std::vector<Index>& ts = ax->tile_sizes;
        if(ts.size() == 1)
        {
            basetile[i] = ts[0];
            continue;
        }
        Index base = ts[0];
        for(size_t t = 1; t < ts.size() - 1; ++t)
        {
            if(ts[t] != base)
            {
                throw std::invalid_argument(
                    "TensorGraph::Runtime: axis " + std::to_string(i) +
                    " of '" + node->name() +
                    "' has unsupported tiling: all tile sizes except the last "
                    "must be equal (got heterogeneous sizes); NNTile supports "
                    "only base tile + optional leftover");
            }
        }
        Index last = ts.back();
        if(last <= 0 || last > base)
        {
            throw std::invalid_argument(
                "TensorGraph::Runtime: axis " + std::to_string(i) +
                " of '" + node->name() +
                "' has unsupported tiling: last tile size must be positive "
                "and not greater than the base tile size (got last=" +
                std::to_string(last) + ", base=" + std::to_string(base) + ")");
        }
        basetile[i] = base;
    }
    return basetile;
}

} // namespace nntile::graph
