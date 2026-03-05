/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/graph_data_node.hh
 * TensorGraph::TensorNode - data node for TensorGraph (shape, dtype, name).
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <cstdint>
#include <functional>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/dtype.hh>

namespace nntile::graph
{

// Forward declaration
class TensorGraph;

//! Data node for TensorGraph - represents a tensor in the graph.
class TensorGraph::TensorNode
{
public:
    using NodeId = uint64_t;

    TensorNode(
        NodeId id,
        TensorGraph* graph,
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
    bool is_compatible(const TensorNode* other) const;

    // Graph access
    TensorGraph* graph();
    const TensorGraph* graph() const;

    // Graph structure
    bool is_input() const { return is_input_; }
    bool is_output() const { return is_output_; }
    void mark_input(bool v = true) { is_input_ = v; }
    void mark_output(bool v = true) { is_output_ = v; }

    // Bind hint: data to copy into runtime tensor when Runtime::compile() runs.
    // Data must already be in NNTile (Fortran) layout. Pass-by-value enables
    // move when caller uses std::move().
    void set_bind_hint(std::vector<std::uint8_t> data);
    const std::vector<std::uint8_t>* get_bind_hint() const;

    // String representation
    std::string to_string() const;

private:
    NodeId id_;
    TensorGraph* graph_;
    std::vector<Index> shape_;
    DataType dtype_;
    std::string name_;
    bool is_input_ = false;
    bool is_output_ = false;
    std::optional<std::vector<std::uint8_t>> bind_hint_;

    friend class TensorGraph;
};

} // namespace nntile::graph
