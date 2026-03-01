/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/logical/mask_scalar.cc
 * Logical graph mask_scalar operation.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/logical/mask_scalar.hh"

// Include standard headers
#include <stdexcept>
#include <utility>

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! Mask scalar operation: conditionally set values based on mask
void mask_scalar(
    LogicalGraph::TensorNode& mask,
    LogicalGraph::TensorNode& x,
    Scalar val,
    Index batch_ndim)
{
    if(&mask.graph() != &x.graph())
    {
        throw std::invalid_argument(
            "mask_scalar: tensors must belong to the same graph");
    }

    if(mask.dtype() != DataType::BOOL)
    {
        throw std::invalid_argument(
            "mask_scalar: mask tensor must have bool dtype");
    }

    auto attrs = std::make_shared<MaskScalarAttrs>(MaskScalarAttrs{val, batch_ndim});
    mask.graph().add_op(
        OpType::MASK_SCALAR,
        attrs,
        {&mask, &x},
        {&x}
    );
}

} // namespace nntile::graph
