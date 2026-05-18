/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/sqrt_inplace.cc
 * TensorGraph sqrt_inplace operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/sqrt_inplace.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/graph/tensor/tile_lowering_helpers.hh"
#include "nntile/graph/tile/lowering_context.hh"
#include "nntile/graph/tile/sqrt_inplace.hh"
#include "nntile/tensor/sqrt_inplace.hh"

namespace nntile::graph::tensor
{

void TensorSqrtInplaceOp::lower_to_tile(const LoweringContext& ctx) const
{
    tile_lower::lower_inplace1(
        dst, ctx.tile_map, "SQRT_INPLACE", tile_graph::sqrt_inplace);
}

void sqrt_inplace(TensorGraph::TensorNode* dst)
{
    if(dst == nullptr)
    {
        throw std::invalid_argument("sqrt_inplace: input tensor must be non-null");
    }

    auto op = std::make_shared<TensorSqrtInplaceOp>(dst);
    dst->graph()->add_op(op);
}

} // namespace nntile::graph::tensor
