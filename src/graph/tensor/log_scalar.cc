/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/log_scalar.cc
 * TensorGraph log_scalar operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/log_scalar.hh"

#include <stdexcept>

#include "nntile/graph/tensor.hh"
#include "nntile/graph/tensor/tile_lowering_helpers.hh"
#include "nntile/graph/tile/log_scalar.hh"
#include "nntile/graph/tile/lowering_context.hh"
#include "nntile/tensor/log_scalar.hh"

namespace nntile::graph::tensor
{

void TensorLogScalarOp::lower_to_tile(const LoweringContext& ctx) const
{
    // Match nntile::tensor::log_scalar_async (src/tensor/log_scalar.cc).
    const auto& tiles = tile_lower::tiles_of(ctx.tile_map, value);
    if(tiles.size() != 1)
    {
        throw std::runtime_error(
            "lower_to_tile LOG_SCALAR: value must be single-tile scalar tensor");
    }
    tile_graph::log_scalar(name, tiles[0]);
}

void log_scalar(const std::string& name,
                TensorGraph::TensorNode* value)
{
    if(value == nullptr)
        throw std::invalid_argument("log_scalar: value tensor must be non-null");
    auto op = std::make_shared<TensorLogScalarOp>(name, value);
    value->graph()->add_op(op);
}

} // namespace nntile::graph::tensor
