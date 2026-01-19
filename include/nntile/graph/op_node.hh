/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/op_node.hh
 * OpNode class for logical graph operation nodes.
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/graph/tensor_node.hh>
#include <nntile/base_types.hh>
#include <string>
#include <vector>
#include <variant>
#include <map>

namespace nntile::graph
{

//! Operation types
enum class OpType {
    GEMM,
    GELU
    // Add more as needed
};

//! Convert OpType to string
std::string op_type_to_string(OpType type);

//! Operation attributes (parameters that aren't tensors)
struct GemmAttrs {
    bool trans_a = false;
    bool trans_b = false;
    // For GEMM: C = alpha * A @ B + beta * C
    Scalar alpha = 1.0;
    Scalar beta = 0.0;
    Index ndim = 1;  // Number of dimensions used in gemm contraction
    Index batch_ndim = 0;  // Number of last dimensions used for batching
};

struct GeluAttrs {
    // No attributes for basic gelu
};

using OpAttrs = std::variant<GemmAttrs, GeluAttrs>;

//! An operation node in the logical graph
class OpNode {
    friend class LogicalGraph;

private:
    NodeId id_;
    OpType type_;
    OpAttrs attrs_;
    LogicalGraph* graph_;

    // Graph edges
    std::vector<TensorNode*> inputs_;
    std::vector<TensorNode*> outputs_;

public:
    OpNode(NodeId id, OpType type, OpAttrs attrs, LogicalGraph* graph);

    // Accessors
    NodeId id() const { return id_; }
    OpType type() const { return type_; }
    const OpAttrs& attrs() const { return attrs_; }

    // Graph structure
    const std::vector<TensorNode*>& inputs() const { return inputs_; }
    const std::vector<TensorNode*>& outputs() const { return outputs_; }

    // Convenience accessors for common cases
    TensorNode* input(size_t idx = 0) const { return inputs_.at(idx); }
    TensorNode* output(size_t idx = 0) const { return outputs_.at(idx); }

    // String representation
    std::string to_string() const;

private:
    // Only LogicalGraph can modify
    void add_input(TensorNode* t);
    void add_output(TensorNode* t);
};

} // namespace nntile::graph
