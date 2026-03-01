/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/logical/multiply_slice.hh
 * Multiply slice operation.
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <string>

// Include other NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

struct MultiplySliceAttrs
{
    Index axis = 0;
    Scalar alpha = 1.0;
    Scalar beta = 0.0;
};

//! Multiply slice operation: tensor = alpha * tensor * slice (broadcasted along axis)
//! @param alpha Scaling factor
//! @param slice Input slice tensor (ndim dimensions)
//! @param tensor Input/output tensor (ndim+1 dimensions, modified in-place)
//! @param axis Axis in tensor along which slice is broadcasted
void multiply_slice(
    Scalar alpha,
    LogicalGraph::TensorNode& slice,
    LogicalGraph::TensorNode& tensor,
    Index axis
);

} // namespace nntile::graph
