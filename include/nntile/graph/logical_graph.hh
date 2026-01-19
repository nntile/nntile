/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/logical_graph.hh
 * LogicalGraph class for defining computation graphs.
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/graph/tensor_node.hh>
#include <nntile/graph/op_node.hh>
#include <memory>
#include <vector>
#include <map>
#include <set>

namespace nntile::graph {

//! Logical graph - defines computation without physical details
class LogicalGraph {
private:
    std::string name_;
    std::vector<std::unique_ptr<TensorNode>> tensors_;
    std::vector<std::unique_ptr<OpNode>> ops_;
    std::map<std::string, TensorNode*> tensor_by_name_;
    std::set<std::string> output_names_;

    NodeId next_tensor_id_ = 0;
    NodeId next_op_id_ = 0;

public:
    explicit LogicalGraph(const std::string& name = "");

    // ═══════════════════════════════════════════════════════════════
    // Tensor Creation
    // ═══════════════════════════════════════════════════════════════

    //! Create a tensor
    TensorNode& tensor(const TensorSpec& spec, const std::string& name);

    //! Mark tensor as output
    void mark_output(const std::string& name);

    // ═══════════════════════════════════════════════════════════════
    // Operations
    // ═══════════════════════════════════════════════════════════════

    //! General matrix multiplication: C = alpha * A @ B + beta * C
    //! Returns reference to output tensor
    TensorNode& gemm(
        TensorNode& a,
        TensorNode& b,
        const std::string& output_name,
        double alpha = 1.0,
        double beta = 0.0,
        bool trans_a = false,
        bool trans_b = false,
        Index ndim = 1,
        Index batch_ndim = 0
    );

    //! GeLU activation: y = gelu(x)
    TensorNode& gelu(
        TensorNode& x,
        const std::string& output_name
    );

    // ═══════════════════════════════════════════════════════════════
    // Queries
    // ═══════════════════════════════════════════════════════════════

    const std::string& name() const { return name_; }
    size_t num_tensors() const { return tensors_.size(); }
    size_t num_ops() const { return ops_.size(); }

    //! Get tensor by name (returns nullptr if not found)
    TensorNode* get_tensor(const std::string& name);
    const TensorNode* get_tensor(const std::string& name) const;

    //! Get all tensor names
    std::vector<std::string> tensor_names() const;

    //! Get output tensor names
    const std::set<std::string>& output_names() const { return output_names_; }

    //! Check if tensor is an output
    bool is_output(const std::string& name) const;

    //! Get all tensors (for iteration)
    const std::vector<std::unique_ptr<TensorNode>>& tensors() const { return tensors_; }

    //! Get all ops (for iteration)
    const std::vector<std::unique_ptr<OpNode>>& ops() const { return ops_; }

    //! String representation
    std::string to_string() const;

private:
    //! Internal: create output tensor for an operation
    TensorNode& create_op_output(
        OpNode& op,
        const TensorSpec& spec,
        const std::string& name
    );

    //! Compute output shape for gemm
    TensorSpec compute_gemm_output_spec(
        const TensorSpec& a,
        const TensorSpec& b,
        bool trans_a,
        bool trans_b,
        Index ndim,
        Index batch_ndim
    );
};

} // namespace nntile::graph
