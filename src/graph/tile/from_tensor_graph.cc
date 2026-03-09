/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tile/from_tensor_graph.cc
 * TileGraph::from_tensor_graph - convert TensorGraph to TileGraph.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tile/graph.hh"
#include "nntile/graph/tile/graph_ops.hh"

#include "nntile/graph/tensor/graph.hh"
#include "nntile/graph/tensor/graph_ops.hh"

#include <map>
#include <stdexcept>

namespace nntile::graph
{

namespace
{

using TNodeMap = std::map<const TensorGraph::TensorNode*, TileGraph::TileNode*>;

void lower_add(const TensorGraph::OpNode& op, TileGraph& tg, const TNodeMap& m)
{
    const auto& add_op = static_cast<const tensor::TensorAddOp&>(op);
    tile_graph::add(add_op.alpha, m.at(add_op.x), add_op.beta,
                    m.at(add_op.y), m.at(add_op.z));
}

void lower_add_inplace(const TensorGraph::OpNode& op, TileGraph& tg,
                       const TNodeMap& m)
{
    const auto& aip = static_cast<const tensor::TensorAddInplaceOp&>(op);
    tile_graph::add_inplace(aip.alpha, m.at(aip.x), aip.beta, m.at(aip.y));
}

void lower_fill(const TensorGraph::OpNode& op, TileGraph& tg,
                const TNodeMap& m)
{
    const auto& fop = static_cast<const tensor::TensorFillOp&>(op);
    tile_graph::fill(fop.val, m.at(fop.x));
}

void lower_clear(const TensorGraph::OpNode& op, TileGraph& tg,
                 const TNodeMap& m)
{
    const auto& cop = static_cast<const tensor::TensorClearOp&>(op);
    tile_graph::clear(m.at(cop.x));
}

} // namespace

TileGraph TileGraph::from_tensor_graph(const TensorGraph& tg)
{
    TileGraph tile_graph(tg.name() + "_tile");

    TNodeMap node_map;

    for(const auto& tensor_node : tg.tensor_nodes())
    {
        const std::vector<Index>& tensor_shape = tensor_node->shape();

        // For now: 1 tile per tensor, tile_shape == tensor_shape
        TileNode* tile_node = tile_graph.data(
            tensor_shape,
            tensor_node->name(),
            tensor_node->dtype());

        if(tensor_node->is_input())
        {
            tile_node->mark_input(true);
        }
        if(tensor_node->is_output())
        {
            tile_node->mark_output(true);
        }

        TensorDescriptor desc;
        desc.tensor_name = tensor_node->name();
        desc.tensor_shape = tensor_shape;
        desc.tile_shape = tensor_shape;
        desc.grid_shape.assign(tensor_shape.size(), 1);
        desc.dtype = tensor_node->dtype();
        desc.tiles = {tile_node};
        desc.source_node = tensor_node.get();

        TensorDescriptor* desc_ptr = tile_graph.add_tensor_descriptor(
            std::move(desc));

        std::vector<Index> coord(tensor_shape.size(), 0);
        tile_node->set_tensor_info(desc_ptr, std::move(coord));

        node_map[tensor_node.get()] = tile_node;
    }

    for(const auto& op : tg.ops())
    {
        const std::string& oname = op->op_name();

        if(oname == "ADD")
        {
            lower_add(*op, tile_graph, node_map);
        }
        else if(oname == "ADD_INPLACE")
        {
            lower_add_inplace(*op, tile_graph, node_map);
        }
        else if(oname == "FILL")
        {
            lower_fill(*op, tile_graph, node_map);
        }
        else if(oname == "CLEAR")
        {
            lower_clear(*op, tile_graph, node_map);
        }
        else
        {
            throw std::runtime_error(
                "TileGraph::from_tensor_graph: unsupported op '" + oname +
                "'; add a lowering rule for this operation");
        }
    }

    return tile_graph;
}

} // namespace nntile::graph
