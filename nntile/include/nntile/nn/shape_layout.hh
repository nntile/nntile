/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/nn/shape_layout.hh
 * NNGraph shape layout helpers (aliases to tensor::shape_layout).
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tensor/shape_layout.hh>

namespace nntile::nn
{

using tensor::graph_axis_to_storage;
using tensor::graph_shape_to_storage;
using tensor::reverse_shape;
using tensor::storage_axis_to_graph;
using tensor::storage_shape_to_graph;

} // namespace nntile::nn
