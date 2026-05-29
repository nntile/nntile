/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/tensor_graph/gelutanh_inplace.cc
 * TensorGraph gelutanh_inplace operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/ops/gelutanh_inplace.hh"

#include <stdexcept>

#include "nntile/dtype.hh"
#include "nntile/tensor.hh"
#include "nntile/tensor/ops/gelutanh_inplace.hh"

#include <nntile/tile/graph_ops.hh>
#include <nntile/tensor/tile_lowering_helpers.hh>

namespace nntile::tensor
{



void gelutanh_inplace(TensorGraph::TensorNode* dst)
{
    if(dst == nullptr)
    {
        throw std::invalid_argument(
            "gelutanh_inplace: dst tensor must be non-null");
    }

    auto op = std::make_shared<TensorGelutanhInplaceOp>(dst);
    dst->graph()->add_op(op);
}

void TensorGelutanhInplaceOp::lower_to_tile(const LoweringContext& ctx) const
{
    tile_lower::lower_inplace1(dst, ctx.tile_map, "GELUTANH_INPLACE",
        tile_graph::gelutanh_inplace);
}

} // namespace nntile::tensor
