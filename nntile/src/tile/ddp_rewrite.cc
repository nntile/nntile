#include <nntile/common.hh>
/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file nntile/src/tile/ddp_rewrite.cc
 *
 * @version 1.1.0
 * */

#include "nntile/tile/ddp.hh"

#include "nntile/core/execution_worker.hh"
#include "nntile/tensor/axis_descriptor.hh"
#include "nntile/tensor/graph.hh"
#include "nntile/tensor/graph_data_node.hh"
#include "nntile/tile/graph.hh"
#include "nntile/tile/ops/add_inplace.hh"
#include "nntile/tile/ops/clear.hh"
#include "nntile/tile/ops/conv2d_bwd_weight_inplace.hh"
#include "nntile/tile/ops/embedding_backward.hh"
#include "nntile/tile/ops/gemm.hh"
#include "nntile/tile/ops/maxsumexp.hh"
#include "nntile/tile/ops/sum.hh"
#include "nntile/tile/ops/sum_fiber.hh"
#include "nntile/tile/ops/sum_slice.hh"
#include "nntile/tile/ops/sumprod_fiber.hh"
#include "nntile/tile/ops/sumprod_slice.hh"

#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace nntile::tile
{

namespace
{

int tile_ddp_replica(
    TileGraph::TileNode const *tile,
    std::string const &axis)
{
    if (tile == nullptr)
    {
        return -1;
    }
    TileGraph::TensorDescriptor const *desc =
        tile->tensor_descriptor();
    if (desc == nullptr || desc->source_node == nullptr)
    {
        return -1;
    }
    TensorGraph::TensorNode const *src = desc->source_node;
    std::vector<Index> const &coord = tile->tile_coord();
    for (int d = 0; d < src->ndim(); ++d)
    {
        AxisDescriptor const *ad = src->axis(d);
        if (ad == nullptr || ad->name != axis)
        {
            continue;
        }
        if (static_cast<size_t>(d) >= coord.size())
        {
            return -1;
        }
        return static_cast<int>(coord[static_cast<size_t>(d)]);
    }
    return -1;
}

bool set_accumulate_beta_zero(TileGraph::OpNode &op)
{
    if (auto *p = dynamic_cast<TileGemmOp *>(&op))
    {
        p->beta = 0;
        return true;
    }
    if (auto *p = dynamic_cast<TileSumOp *>(&op))
    {
        p->beta = 0;
        return true;
    }
    if (auto *p = dynamic_cast<TileSumSliceOp *>(&op))
    {
        p->beta = 0;
        return true;
    }
    if (auto *p = dynamic_cast<TileSumFiberOp *>(&op))
    {
        p->beta = 0;
        return true;
    }
    if (auto *p = dynamic_cast<TileSumprodSliceOp *>(&op))
    {
        p->beta = 0;
        return true;
    }
    if (auto *p = dynamic_cast<TileSumprodFiberOp *>(&op))
    {
        p->beta = 0;
        return true;
    }
    if (auto *p = dynamic_cast<TileEmbeddingBackwardOp *>(&op))
    {
        p->beta = 0;
        return true;
    }
    if (auto *p = dynamic_cast<TileConv2dBwdWeightInplaceOp *>(&op))
    {
        p->beta = 0;
        return true;
    }
    if (auto *p = dynamic_cast<TileMaxsumexpOp *>(&op))
    {
        p->beta = 0;
        return true;
    }
    return false;
}

int unique_input_replica(
    TileGraph::OpNode const &op,
    std::string const &axis)
{
    int found = -1;
    for (TileGraph::TileNode const *t : op.inputs())
    {
        int const r = tile_ddp_replica(t, axis);
        if (r < 0)
        {
            continue;
        }
        if (found < 0)
        {
            found = r;
        }
        else if (found != r)
        {
            return -2;
        }
    }
    return found;
}

int unique_op_replica(
    TileGraph::OpNode const &op,
    std::string const &axis)
{
    int found = -1;
    auto consider = [&](TileGraph::TileNode const *t) {
        int const r = tile_ddp_replica(t, axis);
        if (r < 0)
        {
            return true;
        }
        if (found < 0)
        {
            found = r;
            return true;
        }
        return found == r;
    };
    for (TileGraph::TileNode const *t : op.inputs())
    {
        if (!consider(t))
        {
            return -2;
        }
    }
    for (TileGraph::TileNode const *t : op.outputs())
    {
        if (!consider(t))
        {
            return -2;
        }
    }
    return found;
}

bool op_writes_tile(
    TileGraph::OpNode const &op,
    TileGraph::TileNode const *tile)
{
    if (tile == nullptr)
    {
        return false;
    }
    for (TileGraph::TileNode const *t : op.outputs())
    {
        if (t == tile)
        {
            return true;
        }
    }
    return false;
}

bool op_reads_tile(
    TileGraph::OpNode const &op,
    TileGraph::TileNode const *tile)
{
    if (tile == nullptr)
    {
        return false;
    }
    for (TileGraph::TileNode const *t : op.inputs())
    {
        if (t == tile)
        {
            return true;
        }
    }
    return false;
}

std::string local_tile_name(
    TileGraph::TileNode const *canonical,
    int replica)
{
    std::string base = canonical->name();
    if (base.empty())
    {
        base = "tile@" + std::to_string(
            static_cast<unsigned long long>(canonical->id()));
    }
    return base + "_ddp" + std::to_string(replica);
}

} // namespace

std::vector<Index> ddp_tile_sizes(Index extent, int n_replicas)
{
    if (n_replicas <= 0)
    {
        throw std::invalid_argument(
            "ddp_tile_sizes: n_replicas must be positive");
    }
    if (extent < n_replicas)
    {
        throw std::invalid_argument(
            "ddp_tile_sizes: axis extent (" +
            std::to_string(extent) +
            ") is smaller than replica count (" +
            std::to_string(n_replicas) + ")");
    }
    Index const q = extent / n_replicas;
    Index const r = extent % n_replicas;
    std::vector<Index> sizes(static_cast<size_t>(n_replicas));
    for (int i = 0; i < n_replicas; ++i)
    {
        Index const s = q + (i < r ? Index(1) : Index(0));
        if (s <= 0)
        {
            throw std::invalid_argument(
                "ddp_tile_sizes: zero-size tile");
        }
        sizes[static_cast<size_t>(i)] = s;
    }
    return sizes;
}

void apply_ddp_axis_tiling(
    TensorGraph &tensor_graph,
    std::string const &axis)
{
    if (axis.empty())
    {
        throw std::invalid_argument(
            "apply_ddp_axis_tiling: axis name must be non-empty");
    }
    int n = sched::count_execution_workers();
    if (n <= 1)
    {
        return;
    }
    for (AxisDescriptor *group : tensor_graph.axis_groups())
    {
        if (group == nullptr || group->name != axis)
        {
            continue;
        }
        std::vector<Index> const sizes =
            ddp_tile_sizes(group->extent, n);
        if (group->is_tiled())
        {
            if (group->tile_sizes == sizes)
            {
                continue;
            }
            throw std::runtime_error(
                "apply_ddp_axis_tiling: axis '" + axis +
                "' is already tiled with a different layout");
        }
        group->set_tiling(sizes);
    }
}

void rewrite_ddp_pending(
    TileGraph &tile_graph,
    size_t op_begin,
    size_t op_end,
    std::string const &axis)
{
    if (axis.empty())
    {
        throw std::invalid_argument(
            "rewrite_ddp_pending: axis name must be non-empty");
    }
    if (op_begin > op_end || op_end > tile_graph.num_ops())
    {
        throw std::out_of_range(
            "rewrite_ddp_pending: bad pending op range");
    }
    int n = sched::count_execution_workers();
    if (n <= 0)
    {
        n = 1;
    }

    auto const &ops = tile_graph.ops();
    for (size_t i = op_begin; i < op_end; ++i)
    {
        TileGraph::OpNode &op = *ops[i];
        int const replica = unique_op_replica(op, axis);
        if (replica >= 0)
        {
            op.set_device_hint(replica);
        }
        else
        {
            op.set_device_hint(-1);
        }
    }

    if (n <= 1)
    {
        return;
    }

    struct DestState
    {
        std::map<int, TileGraph::TileNode *> locals;
        std::map<int, size_t> first_write;
        std::map<int, size_t> last_write;
        std::map<int, bool> needs_clear;
    };
    std::unordered_map<TileGraph::TileNode *, DestState> dests;

    for (size_t i = op_begin; i < op_end; ++i)
    {
        TileGraph::OpNode &op = *ops[i];
        int const src_rep = unique_input_replica(op, axis);
        if (src_rep < 0)
        {
            continue;
        }
        std::unordered_set<TileGraph::TileNode *> seen;
        for (TileGraph::TileNode *dest : op.outputs())
        {
            if (dest == nullptr || !seen.insert(dest).second)
            {
                continue;
            }
            if (dest->tensor_descriptor() == nullptr)
            {
                continue;
            }
            if (tile_ddp_replica(dest, axis) >= 0)
            {
                continue;
            }
            DestState &st = dests[dest];
            auto loc_it = st.locals.find(src_rep);
            if (loc_it == st.locals.end())
            {
                TileGraph::TileNode *local = tile_graph.data(
                    dest->shape(),
                    local_tile_name(dest, src_rep),
                    dest->dtype());
                loc_it = st.locals.emplace(src_rep, local).first;
            }
            TileGraph::TileNode *local = loc_it->second;
            op.replace_tile(dest, local);
            if (st.first_write.find(src_rep) == st.first_write.end())
            {
                st.first_write[src_rep] = i;
                st.needs_clear[src_rep] =
                    !set_accumulate_beta_zero(op);
            }
            st.last_write[src_rep] = i;
        }
    }

    if (dests.empty())
    {
        return;
    }

    std::map<size_t, std::vector<std::shared_ptr<TileGraph::OpNode>>,
        std::greater<size_t>>
        inserts;

    for (auto &[canonical, st] : dests)
    {
        size_t last_write = op_begin;
        bool have_write = false;
        for (auto const &[rep, idx] : st.last_write)
        {
            (void)rep;
            if (!have_write || idx > last_write)
            {
                last_write = idx;
                have_write = true;
            }
        }
        if (!have_write)
        {
            continue;
        }
        size_t first_consumer = op_end;
        for (size_t i = op_begin; i < op_end; ++i)
        {
            TileGraph::OpNode const &op = *ops[i];
            if (!op_reads_tile(op, canonical))
            {
                continue;
            }
            bool writes_local = false;
            for (auto const &[rep, local] : st.locals)
            {
                (void)rep;
                if (op_writes_tile(op, local))
                {
                    writes_local = true;
                    break;
                }
            }
            if (writes_local)
            {
                continue;
            }
            first_consumer = i;
            break;
        }
        size_t insert_at = last_write + 1;
        if (first_consumer < insert_at)
        {
            insert_at = first_consumer;
        }
        bool first_add = true;
        for (auto const &[rep, local] : st.locals)
        {
            Scalar const beta = first_add ? Scalar(0) : Scalar(1);
            first_add = false;
            auto add = std::make_shared<TileAddInplaceOp>(
                local, canonical, Scalar(1), beta);
            add->set_device_hint(-1);
            inserts[insert_at].push_back(std::move(add));
            (void)rep;
        }
        for (auto const &[rep, idx] : st.first_write)
        {
            if (!st.needs_clear[rep])
            {
                continue;
            }
            auto clear = std::make_shared<TileClearOp>(
                st.locals.at(rep));
            clear->set_device_hint(rep);
            inserts[idx].push_back(std::move(clear));
        }
    }

    for (auto &[idx, extra] : inserts)
    {
        tile_graph.insert_ops(idx, std::move(extra));
    }
}

} // namespace nntile::tile
