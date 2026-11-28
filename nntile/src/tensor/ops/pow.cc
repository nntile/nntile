/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file nntile/src/tensor_graph/pow.cc
 * TensorGraph pow operation implementation.
 *
 * @version 2.0.0
 * */
/*!
 * INTENTIONALLY DISABLED in the NNTile 2.0 transformer-only scope.
 * Kept in-tree for future re-enablement. No product path calls this code;
 * its tests are not registered while the family is in
 * NNTILE_DISABLED_KERNELS.
 */
#include "nntile/tensor/ops/pow.hh"
#include "nntile/kernel_scope.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/dtype.hh"
#include "nntile/tensor.hh"
#include "nntile/tensor/ops/pow.hh"

#include <nntile/tile/graph_ops.hh>
#include <nntile/tensor/tile_lowering_helpers.hh>

namespace nntile::tensor
{



void pow(
    Scalar alpha,
    Scalar exp,
    TensorGraph::TensorNode* A)
{
    nntile::require_kernel_family("pow");
    if(A == nullptr)
    {
        throw std::invalid_argument(
            "pow: input tensor must be non-null");
    }

    auto op = std::make_shared<TensorPowOp>(alpha, exp, A);
    A->graph()->add_op(op);
}

void TensorPowOp::lower_to_tile(const LoweringContext& ctx) const
{
    for(TileGraph::TileNode* t : tile_lower::tiles_of(ctx.tile_map, A))
    {
        tile::pow(alpha, exp, t);
    }
}

} // namespace nntile::tensor
