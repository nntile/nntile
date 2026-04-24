/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/mask_scalar.cc
 * TensorGraph mask_scalar operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/mask_scalar.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/mask_scalar.hh"

#include "nntile/graph/tile/lowering_context.hh"
#include "nntile/graph/tile/mask_scalar.hh"
#include "nntile/graph/tensor/tensor_graph_tiling.hh"
#include "nntile/graph/tensor/tile_lowering_helpers.hh"

namespace nntile::graph::tensor
{



void mask_scalar(TensorGraph::TensorNode* mask,
                 Scalar val,
                 TensorGraph::TensorNode* A,
                 Index batch_ndim)
{
    if(mask == nullptr || A == nullptr)
        throw std::invalid_argument("mask_scalar: tensors must be non-null");
    if(mask->graph() != A->graph())
        throw std::invalid_argument("mask_scalar: tensors must belong to same graph");
    if(mask->dtype() != DataType::BOOL)
        throw std::invalid_argument("mask_scalar: mask must have BOOL dtype");
    Index A_data_ndim = A->ndim() - batch_ndim;
    if(mask->ndim() != A_data_ndim)
    {
        throw std::invalid_argument(
            "mask_scalar: mask.ndim must equal A.ndim - batch_ndim (" +
            std::to_string(mask->ndim()) + " vs " +
            std::to_string(A_data_ndim) + ")");
    }
    for(Index i = 0; i < A_data_ndim; ++i)
    {
        if(mask->shape()[i] != A->shape()[i])
        {
            throw std::invalid_argument(
                "mask_scalar: mask.dim[" + std::to_string(i) +
                "] must match A.dim[" + std::to_string(i) + "] (" +
                std::to_string(mask->shape()[i]) + " vs " +
                std::to_string(A->shape()[i]) + ")");
        }
        merge_axis(mask->mutable_axes()[i], A->mutable_axes()[i]);
    }

    auto op = std::make_shared<TensorMaskScalarOp>(mask, val, A, batch_ndim);
    A->graph()->add_op(op);
}

void TensorMaskScalarOp::lower_to_tile(const LoweringContext& ctx) const
{
    // Match nntile::tensor::mask_scalar_async (src/tensor/mask_scalar.cc).
    const TensorAxisLayout* lay_a = ctx.tiling.find(A);
    const TensorAxisLayout* lay_m = ctx.tiling.find(mask);
    if(lay_a == nullptr || lay_m == nullptr)
    {
        throw std::runtime_error(
            "lower_to_tile MASK_SCALAR: missing tiling for A and/or mask");
    }

    const auto& tiles_mask = tile_lower::tiles_of(ctx.tile_map, mask);
    const auto& tiles_a = tile_lower::tiles_of(ctx.tile_map, A);

    const Index mask_ndim = mask->ndim();
    std::vector<Index> a_coord;
    std::vector<Index> mask_coord(static_cast<size_t>(mask_ndim));

    for(Index lin_a = 0; lin_a < lay_a->grid_volume(); ++lin_a)
    {
        lay_a->grid_coord_from_linear(lin_a, a_coord);
        for(Index j = 0; j < mask_ndim; ++j)
        {
            mask_coord[static_cast<size_t>(j)] =
                a_coord[static_cast<size_t>(j)];
        }
        const Index lin_m = lay_m->grid_linear(mask_coord);
        tile_graph::mask_scalar(
            tiles_mask[static_cast<size_t>(lin_m)],
            val,
            tiles_a[static_cast<size_t>(lin_a)],
            batch_ndim);
    }
}

} // namespace nntile::graph::tensor
