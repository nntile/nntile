/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/tensor_graph/invalidate.cc
 * TensorGraph async invalidate implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/ops/invalidate.hh"

#include "nntile/tensor/graph_data_node.hh" // check_tensor_id_slot

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "nntile/tensor.hh"
#include "nntile/tensor/tensor_ref.hh"
#include "nntile/tile/graph_ops.hh"
#include "nntile/tensor/tile_lowering_helpers.hh"

namespace nntile::tensor
{

namespace
{

//! [C1 diag] env-gated compile sub-pass brackets (NNTILE_COMPILE_BREAKDOWN).
bool compile_breakdown_enabled()
{
    return env_flag_enabled("NNTILE_COMPILE_BREAKDOWN");
}

double breakdown_ms_since(std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
}

} // namespace

void invalidate(TensorGraph::TensorNode *x)
{
    if (x == nullptr)
    {
        throw std::invalid_argument(
            "invalidate: tensor must be non-null");
    }

    auto op = nntile::make_op<TensorInvalidateOp>(x);
    x->graph()->add_op(op);
}

void TensorInvalidateOp::lower_to_tile(const LoweringContext &ctx) const
{
    for (TileGraph::TileNode *t : tile_lower::tiles_of(ctx.tile_map, x))
    {
        tile::invalidate(t);
    }
}

std::size_t append_invalidates_for_unmarked_unsealed(TensorGraph &graph)
{
    bool const cbrk = compile_breakdown_enabled();
    auto const t_scan = std::chrono::steady_clock::now();
    // Dense bitmaps over TensorNode ids (id == data_ slot index, high-water
    // with GC holes). Replaces the two unordered_set<TensorNode*> hash
    // structures whose insert cost grew with graph size (capacity
    // superlinearity). The append order changes from hash order to
    // first-touch order - the appended INVALIDATE ops touch disjoint
    // tensors, so execution semantics are unchanged.
    const auto &ops = graph.ops();
    const std::size_t begin = graph.phase_seal_cursor();
    const std::size_t id_space = graph.num_data();
    std::vector<std::uint8_t> touched(id_space, 0);
    std::vector<std::uint8_t> already_reclaimed(id_space, 0);
    std::vector<TensorGraph::TensorNode *> touched_order;
    touched_order.reserve((ops.size() - begin) + 8);
    auto note_touched = [&](TensorGraph::TensorNode *t)
    {
        if (t == nullptr)
        {
            return;
        }
        check_tensor_id_slot(static_cast<std::uint64_t>(t->id()),
            id_space, "invalidate note_touched");
        std::size_t const id = static_cast<std::size_t>(t->id());
        if (!touched[id])
        {
            touched[id] = 1;
            touched_order.push_back(t);
        }
    };
    for (std::size_t i = begin; i < ops.size(); ++i)
    {
        std::shared_ptr<TensorGraph::OpNode> const &op = ops[i];
        if (op == nullptr)
        {
            continue;
        }

        for (TensorGraph::TensorNode *in : op->inputs())
        {
            note_touched(in);
        }
        for (TensorGraph::TensorNode *ot : op->outputs())
        {
            note_touched(ot);
        }
        if (op->is_unregister())
        {
            for (TensorGraph::TensorNode *in : op->inputs())
            {
                if (in != nullptr)
                {
                    check_tensor_id_slot(
                        static_cast<std::uint64_t>(in->id()), id_space,
                        "invalidate reclaim");
                    already_reclaimed[
                        static_cast<std::size_t>(in->id())] = 1;
                }
            }
        }
    }
    std::size_t const touched_count = touched_order.size();

    double const scan_ms = cbrk ? breakdown_ms_since(t_scan) : 0.0;
    auto const t_append = std::chrono::steady_clock::now();
    // Snapshot BEFORE the append loop: invalidate() grows graph.ops(),
    // so ops.size() - begin at print time would over-count by n_added.
    std::size_t const phase_ops = ops.size() - begin;
    std::size_t n_added = 0;
    for (TensorGraph::TensorNode *t : touched_order)
    {
        if (tensor_ref_is_live(t)
            || already_reclaimed[static_cast<std::size_t>(t->id())])
        {
            continue;
        }
        invalidate(t);
        ++n_added;
    }
    if (cbrk)
    {
        std::fprintf(stderr,
            "[nntile-cbrk] inval phase_ops=%zu scan_ms=%.3f "
            "append_ms=%.3f touched=%zu added=%zu\n",
            phase_ops, scan_ms, breakdown_ms_since(t_append),
            touched_count, n_added);
    }
    return n_added;
}

} // namespace nntile::tensor
