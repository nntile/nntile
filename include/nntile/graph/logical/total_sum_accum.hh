/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/logical/total_sum_accum.hh
 * Logical graph total_sum_accum operation.
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <string>

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! Total sum accumulation: val = alpha * sum(logsumexp * src) + beta * val
//! @param logsumexp Log-sum-exp tensor
//! @param src Source tensor
//! @param class_labels Class labels tensor (int64)
//! @param val Output value tensor (fp32)
//! @param alpha Scaling factor (default: 1.0)
//! @param ignore_index Index to ignore (default: -1)
void total_sum_accum(
    LogicalGraph::TensorNode& logsumexp,
    LogicalGraph::TensorNode& src,
    LogicalGraph::TensorNode& class_labels,
    LogicalGraph::TensorNode& val,
    Scalar alpha = 1.0,
    Index ignore_index = -1
);

} // namespace nntile::graph
