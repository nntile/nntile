/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/phase_codec.hh
 * TensorGraph phase codec: Flush ``PhaseIR`` JSON ``{nodes, ops}``.
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tensor/graph_decl.hh>

#include <nlohmann/json.hpp>

#include <string>
#include <utility>

namespace nntile::tensor
{

//! Flush PhaseIR v1 ``op_name`` allowlist. Unknown names fail closed
//! (``UnknownOp``). Wire names: FILL, COPY, SCATTER, GATHER, INVALIDATE,
//! ADD, MUL, MM, LINEAR, RELU, CROSS_ENTROPY.
inline constexpr char const *kV1PhaseOps[] = {
    "FILL",
    "COPY",
    "SCATTER",
    "GATHER",
    "INVALIDATE",
    "ADD",
    "MUL",
    "MM",
    "LINEAR",
    "RELU",
    "CROSS_ENTROPY",
};

bool is_v1_phase_op(std::string const &op_name);

//! Validate ``ops`` and return ``{"nodes": nodes, "ops": ops}``.
//! Null ``nodes`` / ``ops`` become empty arrays.
nlohmann::json encode_phase(
    nlohmann::json const &nodes,
    nlohmann::json const &ops);

//! Encode unsealed ops ``[phase_seal_cursor, num_ops)`` and the nodes
//! those ops reference. Does not seal. TensorGraph ``MULTIPLY`` becomes
//! wire ``MUL``; ``GEMM`` becomes ``MM``.
nlohmann::json encode_phase(TensorGraph const &graph);

//! Encode one snapshot slice of ``graph.ops()``.
nlohmann::json encode_phase(
    TensorGraph const &graph,
    TensorGraph::PhaseSnapshot const &phase);

//! Unpack Flush PhaseIR. Unknown ``op_name`` throws ``UnknownOp``.
std::pair<nlohmann::json, nlohmann::json> decode_phase(
    nlohmann::json const &blob);

} // namespace nntile::tensor
