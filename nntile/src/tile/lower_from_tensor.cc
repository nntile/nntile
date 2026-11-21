/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/tile_graph/lower_from_tensor.cc
 * TileGraph lower from tensor operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/tile/lower_from_tensor.hh"
#include "nntile/tensor/graph_data_node.hh"

#include <nntile/tensor/graph.hh>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace nntile
{

void lower_tensor_ops_to_tile_graph(
    const TensorGraph& tg,
    TileGraph& out,
    const TensorNodeToTileMap& tile_map)
{
    lower_tensor_ops_to_tile_graph(tg, out, tile_map, 0, tg.ops().size());
}

void lower_tensor_ops_to_tile_graph(
    const TensorGraph& tg,
    TileGraph& out,
    const TensorNodeToTileMap& tile_map,
    size_t op_begin,
    size_t op_end)
{
    const TensorGraphTiling* tsch = out.tiling_scheme();
    if(tsch == nullptr)
    {
        throw std::runtime_error(
            "lower_tensor_ops_to_tile_graph: TileGraph has no tiling_scheme");
    }

    const auto& ops = tg.ops();
    if(op_end > ops.size())
    {
        throw std::out_of_range(
            "lower_tensor_ops_to_tile_graph: op_end > num_ops");
    }
    if(op_begin > op_end)
    {
        throw std::invalid_argument(
            "lower_tensor_ops_to_tile_graph: op_begin > op_end");
    }

    const LoweringContext ctx{out, tile_map, *tsch};
    // Distance prefetch: each op's lowering probes this op's tensors'
    // axis layouts (dense-by-id optionals) and tile lists (dense-by-id
    // slots) with random ids - pure latency at sizes past LLC. Warm the
    // probes a few ops ahead so the geometry compute of op i overlaps
    // the misses of ops i+PREFETCH_OPS..i+2*PREFETCH_OPS.
    constexpr std::size_t kPrefetchOps = 6;
    auto prefetch_slots = [&](size_t i)
    {
        if (i >= op_end)
        {
            return;
        }
        TensorGraph::OpNode const &op = *ops[i];
        for (TensorGraph::TensorNode const *t : op.inputs())
        {
            if (t != nullptr)
            {
                __builtin_prefetch(tsch->find_slot(t), 0, 1);
                __builtin_prefetch(tile_map.try_get_slot(t), 0, 1);
            }
        }
        for (TensorGraph::TensorNode const *t : op.outputs())
        {
            if (t != nullptr)
            {
                __builtin_prefetch(tsch->find_slot(t), 0, 1);
                __builtin_prefetch(tile_map.try_get_slot(t), 0, 1);
            }
        }
    };
    // Stage 2: the slot prefetch above made the layout objects resident
    // by now; warm their flat geometry buffers too.
    auto prefetch_data = [&](size_t i)
    {
        if (i >= op_end)
        {
            return;
        }
        TensorGraph::OpNode const &op = *ops[i];
        for (TensorGraph::TensorNode const *t : op.inputs())
        {
            if (t != nullptr)
            {
                tsch->prefetch_layout_data(t);
            }
        }
        for (TensorGraph::TensorNode const *t : op.outputs())
        {
            if (t != nullptr)
            {
                tsch->prefetch_layout_data(t);
            }
        }
    };
    for (size_t k = 1; k <= 2 * kPrefetchOps; ++k)
    {
        prefetch_slots(op_begin + k);
    }
    for (size_t k = 1; k <= kPrefetchOps; ++k)
    {
        prefetch_data(op_begin + k);
    }
    for(size_t i = op_begin; i < op_end; ++i)
    {
        prefetch_slots(i + 2 * kPrefetchOps);
        prefetch_data(i + kPrefetchOps);
        ops[i]->lower_to_tile(ctx);
    }
}

} // namespace nntile
