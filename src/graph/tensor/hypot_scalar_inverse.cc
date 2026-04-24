/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/hypot_scalar_inverse.cc
 * TensorGraph hypot_scalar_inverse operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/hypot_scalar_inverse.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/graph/tensor/tile_lowering_helpers.hh"
#include "nntile/graph/tile/hypot_scalar_inverse.hh"
#include "nntile/graph/tile/lowering_context.hh"
#include "nntile/tensor/hypot_scalar_inverse.hh"

namespace nntile::graph::tensor
{

void TensorHypotScalarInverseOp::lower_to_tile(const LoweringContext& ctx) const
{
    for(TileGraph::TileNode* t : tile_lower::tiles_of(ctx.tile_map, dst))
    {
        tile_graph::hypot_scalar_inverse(eps, alpha, t);
    }
}

void hypot_scalar_inverse(
    Scalar eps,
    Scalar alpha,
    TensorGraph::TensorNode* dst)
{
    if(dst == nullptr)
    {
        throw std::invalid_argument(
            "hypot_scalar_inverse: dst tensor must be non-null");
    }

    auto op = std::make_shared<TensorHypotScalarInverseOp>(eps, alpha, dst);
    dst->graph()->add_op(op);
}

} // namespace nntile::graph::tensor
