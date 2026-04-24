/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/gelu_inplace.cc
 * TensorGraph gelu_inplace operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/gelu_inplace.hh"

#include <stdexcept>

#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/gelu_inplace.hh"

#include <nntile/graph/tile/graph_ops.hh>
#include <nntile/graph/tensor/tile_lowering_helpers.hh>

namespace nntile::graph::tensor
{



void gelu_inplace(TensorGraph::TensorNode* dst)
{
    if(dst == nullptr)
    {
        throw std::invalid_argument(
            "gelu_inplace: dst tensor must be non-null");
    }

    auto op = std::make_shared<TensorGeluInplaceOp>(dst);
    dst->graph()->add_op(op);
}

void TensorGeluInplaceOp::lower_to_tile(const LoweringContext& ctx) const
{
    tile_lower::lower_inplace1(
        dst, ctx.tile_map, "GELU_INPLACE", tile_graph::gelu_inplace);
}

} // namespace nntile::graph::tensor
