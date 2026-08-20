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

#include <stdexcept>
#include <string>
#include <unordered_set>

#include "nntile/tensor.hh"
#include "nntile/tensor/tensor_ref.hh"
#include "nntile/tile/graph_ops.hh"
#include "nntile/tensor/tile_lowering_helpers.hh"

namespace nntile::tensor
{

void invalidate(TensorGraph::TensorNode *x)
{
    if (x == nullptr)
    {
        throw std::invalid_argument(
            "invalidate: tensor must be non-null");
    }

    auto op = std::make_shared<TensorInvalidateOp>(x);
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
    std::unordered_set<TensorGraph::TensorNode *> touched;
    std::unordered_set<TensorGraph::TensorNode *> already_reclaimed;
    const auto &ops = graph.ops();
    const std::size_t begin = graph.phase_seal_cursor();
    touched.reserve((ops.size() - begin) * 4 + 8);
    for (std::size_t i = begin; i < ops.size(); ++i)
    {
        std::shared_ptr<TensorGraph::OpNode> const &op = ops[i];
        if (op == nullptr)
        {
            continue;
        }
        for (TensorGraph::TensorNode *in : op->inputs())
        {
            if (in != nullptr)
            {
                touched.insert(in);
            }
        }
        for (TensorGraph::TensorNode *ot : op->outputs())
        {
            if (ot != nullptr)
            {
                touched.insert(ot);
            }
        }
        std::string const name = op->op_name();
        if (name == "UNREGISTER")
        {
            for (TensorGraph::TensorNode *in : op->inputs())
            {
                if (in != nullptr)
                {
                    already_reclaimed.insert(in);
                }
            }
        }
    }

    std::size_t n_added = 0;
    for (TensorGraph::TensorNode *t : touched)
    {
        if (t == nullptr || tensor_ref_is_live(t)
            || already_reclaimed.count(t) != 0)
        {
            continue;
        }
        invalidate(t);
        ++n_added;
    }
    return n_added;
}

} // namespace nntile::tensor
