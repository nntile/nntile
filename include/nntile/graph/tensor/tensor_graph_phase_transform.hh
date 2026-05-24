/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/tensor_graph_phase_transform.hh
 * Optional ``TensorGraph`` rewrite between ``NNGraph::finish_phase`` and tile
 * lowering (identity baseline today; future DDP/FSDP-style graphs).
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <functional>

// NNTile headers
#include <nntile/graph/tensor/graph_decl.hh>

namespace nntile::graph
{

//! Baseline transform: lower the finished phase unchanged.
inline FinishedTensorPhase tensor_graph_identity_phase(
    FinishedTensorPhase const& phase)
{
    return phase;
}

//! Rewrites a finished tensor phase before tiling (must preserve marked I/O).
using TensorGraphPhaseTransform =
    std::function<FinishedTensorPhase(FinishedTensorPhase const&)>;

} // namespace nntile::graph
