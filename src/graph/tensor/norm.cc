/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/norm.cc
 * TensorGraph norm operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/norm.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/norm.hh"

#include "nntile/graph/tile/lowering_context.hh"
#include "nntile/graph/tile/norm.hh"
#include "nntile/graph/tensor/tile_lowering_helpers.hh"

namespace nntile::graph::tensor
{



void norm(TensorGraph::TensorNode* x, TensorGraph::TensorNode* y,
          Scalar alpha, Scalar beta)
{
    if(x == nullptr || y == nullptr)
    {
        throw std::invalid_argument("norm: input tensors must be non-null");
    }
    if(x == y)
    {
        throw std::invalid_argument(
            "norm: x and y must be distinct tensors");
    }
    if(x->graph() != y->graph())
    {
        throw std::invalid_argument(
            "norm: tensors must belong to the same graph");
    }
    if(y->ndim() != 0)
    {
        throw std::invalid_argument(
            "norm: output tensor must be scalar (shape [])");
    }
    if(x->dtype() != y->dtype())
    {
        throw std::invalid_argument(
            "norm: input and output tensors must have the same dtype");
    }

    auto op = std::make_shared<TensorNormOp>(x, y, alpha, beta);
    x->graph()->add_op(op);
}

void TensorNormOp::lower_to_tile(const LoweringContext& ctx) const
{
    // Match nntile::tensor::norm_async (src/tensor/norm.cc).
    const auto& tiles_x = tile_lower::tiles_of(ctx.tile_map, x);
    const auto& tiles_y = tile_lower::tiles_of(ctx.tile_map, y);
    if(tiles_y.size() != 1)
    {
        throw std::runtime_error("lower_to_tile NORM: scalar output must be one tile");
    }
    constexpr Scalar one = 1.0;
    for(size_t i = 0; i < tiles_x.size(); ++i)
    {
        if(i == 0)
        {
            tile_graph::norm(alpha, tiles_x[i], beta, tiles_y[0]);
        }
        else
        {
            tile_graph::norm(alpha, tiles_x[i], one, tiles_y[0]);
        }
    }
}

} // namespace nntile::graph::tensor
