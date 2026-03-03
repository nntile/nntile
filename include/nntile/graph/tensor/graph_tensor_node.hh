/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/graph_tensor_node.hh
 * TensorGraphNode - data node for TensorGraph (shape, dtype, name).
 *
 * @version 1.1.0
 * */

#pragma once

#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <cstdint>

#include <nntile/base_types.hh>
#include <nntile/graph/dtype.hh>

namespace nntile::graph
{

//! Data node for TensorGraph - represents a tensor in the graph.
class TensorGraphNode
{
public:
    using NodeId = uint64_t;

    TensorGraphNode(
        NodeId id,
        class TensorGraph* graph,
        std::vector<Index> shape,
        DataType dtype,
        const std::string& name = "");

    // Accessors
    NodeId id() const { return id_; }
    const std::string& name() const { return name_; }
    DataType dtype() const { return dtype_; }
    const std::vector<Index>& shape() const { return shape_; }
    Index ndim() const { return static_cast<Index>(shape_.size()); }
    Index dim(int idx) const;
    Index nelems() const;
    size_t size_bytes() const;
    bool is_compatible(const TensorGraphNode* other) const;

    // Graph access
    class TensorGraph* graph();
    const class TensorGraph* graph() const;

    // Graph structure
    bool is_input() const { return is_input_; }
    bool is_output() const { return is_output_; }
    void mark_input(bool v = true) { is_input_ = v; }
    void mark_output(bool v = true) { is_output_ = v; }

    // String representation
    std::string to_string() const;

private:
    NodeId id_;
    class TensorGraph* graph_;
    std::vector<Index> shape_;
    DataType dtype_;
    std::string name_;
    bool is_input_ = false;
    bool is_output_ = false;

    friend class TensorGraph;
};

} // namespace nntile::graph
