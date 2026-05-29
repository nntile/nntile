/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/tensor_graph/sqrt_inplace.cc
 * TensorGraph sqrt_inplace operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/tensor_graph/ops/sqrt_inplace.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/dtype.hh"
#include "nntile/tensor.hh"
#include "nntile/tensor_graph/tile_lowering_helpers.hh"
#include "nntile/tile_graph/lowering_context.hh"
#include "nntile/tile_graph/ops/sqrt_inplace.hh"
#include "nntile/tensor/sqrt_inplace.hh"

namespace nntile::tensor_graph
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

} // namespace nntile::tensor_graph
