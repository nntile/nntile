/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/basetile.hh
 * Basetile shape for tensor::Tensor allocation from TensorGraph axis descriptors.
 *
 * @version 1.1.0
 * */

#pragma once

#include <vector>

#include <nntile/base_types.hh>
#include <nntile/graph/tensor/graph_decl.hh>

namespace nntile::graph
{

//! Compute basetile_shape from axis descriptors for TensorTraits / tensor::Tensor.
//! If an axis is not tiled: basetile = full extent.
//! If tiled: basetile = first segment size, with validation for NNTile base+leftover.
std::vector<Index> compute_basetile_shape_for_tensor(
    const TensorGraph::TensorNode* node);

} // namespace nntile::graph
