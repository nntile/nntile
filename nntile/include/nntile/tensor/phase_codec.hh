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

#include <nntile/dtype.hh>
#include <nntile/tensor/graph_decl.hh>
#include <nntile/tensor/tensor_ref.hh>

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nntile::tensor
{

//! Flush PhaseIR allowlist: v1 six names, classic TensorGraph ops,
//! and CUDA-only FLASH names. Unknown names fail closed
//! (``UnknownOp``). Last ``TensorRef`` drop records ``UNREGISTER``.
//! Torch-native aten kind is ``attrs.kind``.
inline constexpr char const *kV1PhaseOps[] = {
    "TORCH_UNARY",
    "TORCH_BINARY",
    "TORCH_TERNARY",
    "GATHER",
    "SCATTER",
    "UNREGISTER",
    "ADAM_STEP",
    "ADAMW_STEP",
    "ADD",
    "ADD_FIBER",
    "ADD_FIBER_INPLACE",
    "ADD_INPLACE",
    "ADD_SLICE",
    "ADD_SLICE_INPLACE",
    "CLEAR",
    "CONCAT",
    "CONTIGUOUS_VIEW",
    "CONV2D_BWD_INPUT_INPLACE",
    "CONV2D_BWD_WEIGHT_INPLACE",
    "CONV2D_INPLACE",
    "COPY",
    "COPY_INTERSECTION",
    "EMBEDDING",
    "EMBEDDING_BACKWARD",
    "FILL",
    "GELU",
    "GELU_BACKWARD",
    "GELU_INPLACE",
    "GELUTANH",
    "GELUTANH_BACKWARD",
    "GELUTANH_INPLACE",
    "GEMM",
    "HYPOT",
    "HYPOT_INPLACE",
    "HYPOT_SCALAR_INVERSE",
    "INVALIDATE",
    "LOG_SCALAR",
    "LOGSUMEXP",
    "MASK_SCALAR",
    "MAXSUMEXP",
    "MULTIPLY",
    "MULTIPLY_FIBER",
    "MULTIPLY_FIBER_INPLACE",
    "MULTIPLY_INPLACE",
    "MULTIPLY_SLICE",
    "NORM",
    "NORM_FIBER",
    "NORM_FIBER_INPLACE",
    "NORM_SLICE",
    "NORM_SLICE_INPLACE",
    "POW",
    "RANDN",
    "RELU",
    "RELU_BACKWARD",
    "RELU_INPLACE",
    "ROPE",
    "ROPE_BACKWARD",
    "SCALE",
    "SCALE_FIBER",
    "SCALE_INPLACE",
    "SCALE_SLICE",
    "SGD_STEP",
    "SILU",
    "SILU_BACKWARD",
    "SILU_INPLACE",
    "SOFTMAX",
    "SOFTMAX_INPLACE",
    "SQRT",
    "SQRT_INPLACE",
    "SUBTRACT_INDEXED_OUTPUTS",
    "SUM",
    "SUM_FIBER",
    "SUM_SLICE",
    "SUMPROD_FIBER",
    "SUMPROD_SLICE",
    "SWAP_TWO_AXES",
    "TOTAL_SUM_ACCUM",
    "TRANSPOSE",
    "FLASH_SDPA_BWD_CUDNN",
    "FLASH_SDPA_FWD_CUDNN",
};

inline constexpr std::size_t kV1PhaseOpCount =
    sizeof(kV1PhaseOps) / sizeof(kV1PhaseOps[0]);

bool is_v1_phase_op(std::string const &op_name);

//! Validate ``ops`` and return ``{"nodes": nodes, "ops": ops}``.
//! Null ``nodes`` / ``ops`` become empty arrays.
nlohmann::json encode_phase(
    nlohmann::json const &nodes,
    nlohmann::json const &ops);

//! Encode unsealed ops ``[phase_seal_cursor, num_ops)`` and the nodes
//! those ops reference. Does not seal.
nlohmann::json encode_phase(TensorGraph const &graph);

//! Encode one snapshot slice of ``graph.ops()``.
nlohmann::json encode_phase(
    TensorGraph const &graph,
    TensorGraph::PhaseSnapshot const &phase);

//! Unpack Flush PhaseIR. Unknown ``op_name`` throws ``UnknownOp``.
std::pair<nlohmann::json, nlohmann::json> decode_phase(
    nlohmann::json const &blob);

//! Shape / dtype / name for one PhaseIR node id.
struct PhaseNodeSpec
{
    std::vector<Index> shape;
    DataType dtype = DataType::FP32;
    std::string name;
};

using PhaseNodeMap = std::unordered_map<std::int64_t, TensorRef>;
using PhaseNodeSpecs = std::unordered_map<std::int64_t, PhaseNodeSpec>;

void parse_phase_nodes(
    nlohmann::json const &nodes,
    PhaseNodeSpecs &specs);

//! Create ``graph.data`` for ``id`` if missing. Honors PhaseIR dtype.
TensorRef ensure_phase_node(
    TensorGraph &graph,
    PhaseNodeMap &refs,
    PhaseNodeSpecs const &specs,
    std::int64_t id);

//! Record one PhaseIR op onto ``graph`` via ``gt::*``.
void record_phase_op(
    TensorGraph &graph,
    nlohmann::json const &op,
    PhaseNodeMap &refs,
    PhaseNodeSpecs const &specs);

//! ``decode_phase``, then ``record_phase_op`` for each op.
void apply_phase(
    TensorGraph &graph,
    nlohmann::json const &blob,
    PhaseNodeMap &refs);

} // namespace nntile::tensor
