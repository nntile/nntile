/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/relu_inplace.cc
 * TensorGraph relu_inplace operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/relu_inplace.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/graph/tensor/tile_lowering_helpers.hh"
#include "nntile/graph/tile/lowering_context.hh"
#include "nntile/graph/tile/relu_inplace.hh"
#include "nntile/tensor/relu_inplace.hh"

namespace nntile::graph::tensor
{

void TensorReluInplaceOp::lower_to_tile(const LoweringContext& ctx) const
{
    tile_lower::lower_inplace1(
        dst, ctx.tile_map, "RELU_INPLACE", tile_graph::relu_inplace);
}

void relu_inplace(TensorGraph::TensorNode* dst)
{
    if(dst == nullptr)
    {
        throw std::invalid_argument("relu_inplace: input tensor must be non-null");
    }

    auto op = std::make_shared<TensorReluInplaceOp>(dst);
    dst->graph()->add_op(op);
}

} // namespace nntile::graph::tensor
