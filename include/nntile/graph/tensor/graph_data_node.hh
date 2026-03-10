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
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/dtype.hh>
#include <nntile/graph/tensor/axis_descriptor.hh>

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

    // Axis descriptors (parallel to shape_, one per dimension)
    AxisDescriptor* axis(int i) const;
    const std::vector<std::shared_ptr<AxisDescriptor>>& axes() const
    {
        return axes_;
    }
    std::vector<std::shared_ptr<AxisDescriptor>>& mutable_axes()
    {
        return axes_;
    }

    //! Replace this node's axes with the provided shared axes,
    //! joining their groups. Sizes must match shape.
    void set_axes(const std::vector<std::shared_ptr<AxisDescriptor>>& axes);

    // Graph structure
    bool is_input() const { return is_input_; }
    bool is_output() const { return is_output_; }
    void mark_input(bool v = true) { is_input_ = v; }
    void mark_output(bool v = true) { is_output_ = v; }

    // Bind hint: data to copy into runtime tensor when Runtime::compile() runs.
    void set_bind_hint(std::vector<std::uint8_t> data);
    const std::vector<std::uint8_t>* get_bind_hint() const;

    // String representation
    std::string to_string() const;

private:
    NodeId id_;
    TensorGraph* graph_;
    std::vector<Index> shape_;
    std::vector<std::shared_ptr<AxisDescriptor>> axes_;
    DataType dtype_;
    std::string name_;
    bool is_input_ = false;
    bool is_output_ = false;
    std::optional<std::vector<std::uint8_t>> bind_hint_;

    friend class TensorGraph;
};

//! Validate same shape and merge axes for two tensors (single loop).
inline void validate_same_shape_and_merge(TensorGraph::TensorNode* a,
                                          TensorGraph::TensorNode* b,
                                          const std::string& op_name)
{
    if(a->ndim() != b->ndim())
    {
        throw std::invalid_argument(
            op_name + ": tensors must have same ndim (" +
            std::to_string(a->ndim()) + " vs " + std::to_string(b->ndim()) +
            ")");
    }
    for(Index i = 0; i < a->ndim(); ++i)
    {
        if(a->shape()[i] != b->shape()[i])
        {
            throw std::invalid_argument(
                op_name + ": tensors must have same shape; mismatch at dimension "
                + std::to_string(i) + " (" + std::to_string(a->shape()[i]) +
                " vs " + std::to_string(b->shape()[i]) + ")");
        }
        merge_axis(a->mutable_axes()[i], b->mutable_axes()[i]);
    }
}

//! Validate slice broadcast shape and merge axes (src ndim-1 into dst).
inline void validate_slice_broadcast_shape_and_merge(
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst,
    Index axis,
    const std::string& op_name)
{
    if(src->ndim() + 1 != dst->ndim())
    {
        throw std::invalid_argument(
            op_name + ": src must have ndim = dst.ndim - 1 (" +
            std::to_string(src->ndim()) + " vs " +
            std::to_string(dst->ndim()) + ")");
    }
    if(axis < 0 || axis >= dst->ndim())
    {
        throw std::invalid_argument(op_name + ": axis out of range");
    }
    int d = 0;
    for(Index i = 0; i < dst->ndim(); ++i)
    {
        if(i == axis)
            continue;
        if(src->shape()[d] != dst->shape()[i])
        {
            throw std::invalid_argument(
                op_name + ": shape mismatch at broadcast dimension " +
                std::to_string(i) + " (src " + std::to_string(d) + ": " +
                std::to_string(src->shape()[d]) + " vs dst: " +
                std::to_string(dst->shape()[i]) + ")");
        }
        merge_axis(src->mutable_axes()[d], dst->mutable_axes()[i]);
        ++d;
    }
}

//! Validate slice reduce shape and merge axes (src ndim+1 into dst).
inline void validate_slice_reduce_shape_and_merge(
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst,
    Index axis,
    const std::string& op_name)
{
    if(src->ndim() != dst->ndim() + 1)
    {
        throw std::invalid_argument(
            op_name + ": dst must have ndim = src.ndim - 1 (" +
            std::to_string(dst->ndim()) + " vs " +
            std::to_string(src->ndim()) + ")");
    }
    if(axis < 0 || axis >= src->ndim())
    {
        throw std::invalid_argument(op_name + ": axis out of range");
    }
    int d = 0;
    for(Index i = 0; i < src->ndim(); ++i)
    {
        if(i == axis)
            continue;
        if(src->shape()[i] != dst->shape()[d])
        {
            throw std::invalid_argument(
                op_name + ": shape mismatch at reduce dimension " +
                std::to_string(i) + " (src: " + std::to_string(src->shape()[i]) +
                " vs dst " + std::to_string(d) + ": " +
                std::to_string(dst->shape()[d]) + ")");
        }
        merge_axis(src->mutable_axes()[i], dst->mutable_axes()[d]);
        ++d;
    }
}

//! Validate fiber broadcast shape and merge axes (fiber 1+batch_ndim into tensor).
inline void validate_fiber_broadcast_shape_and_merge(
    TensorGraph::TensorNode* fiber,
    TensorGraph::TensorNode* tensor,
    Index axis,
    Index batch_ndim,
    const std::string& op_name)
{
    if(fiber->ndim() != 1 + batch_ndim)
    {
        throw std::invalid_argument(
            op_name + ": fiber must have ndim = 1 + batch_ndim (" +
            std::to_string(fiber->ndim()) + " vs " +
            std::to_string(1 + batch_ndim) + ")");
    }
    if(axis < 0 || axis >= tensor->ndim())
    {
        throw std::invalid_argument(op_name + ": axis out of range");
    }
    if(fiber->shape()[0] != tensor->shape()[axis])
    {
        throw std::invalid_argument(
            op_name + ": fiber dim 0 must match tensor dim " +
            std::to_string(axis) + " (" + std::to_string(fiber->shape()[0]) +
            " vs " + std::to_string(tensor->shape()[axis]) + ")");
    }
    merge_axis(fiber->mutable_axes()[0], tensor->mutable_axes()[axis]);
    for(Index i = 0; i < batch_ndim; ++i)
    {
        Index ti = tensor->ndim() - batch_ndim + i;
        if(fiber->shape()[1 + i] != tensor->shape()[ti])
        {
            throw std::invalid_argument(
                op_name + ": fiber dim " + std::to_string(1 + i) +
                " must match tensor dim " + std::to_string(ti) + " (" +
                std::to_string(fiber->shape()[1 + i]) + " vs " +
                std::to_string(tensor->shape()[ti]) + ")");
        }
        merge_axis(fiber->mutable_axes()[1 + i],
                   tensor->mutable_axes()[ti]);
    }
}

//! Validate fiber reduce shape and merge axes (tensor reduced to fiber 1+batch_ndim).
inline void validate_fiber_reduce_shape_and_merge(
    TensorGraph::TensorNode* fiber,
    TensorGraph::TensorNode* tensor,
    Index axis,
    Index batch_ndim,
    const std::string& op_name)
{
    if(fiber->ndim() != 1 + batch_ndim)
    {
        throw std::invalid_argument(
            op_name + ": fiber must have ndim = 1 + batch_ndim (" +
            std::to_string(fiber->ndim()) + " vs " +
            std::to_string(1 + batch_ndim) + ")");
    }
    if(axis < 0 || axis >= tensor->ndim())
    {
        throw std::invalid_argument(op_name + ": axis out of range");
    }
    if(fiber->shape()[0] != tensor->shape()[axis])
    {
        throw std::invalid_argument(
            op_name + ": fiber dim 0 must match tensor dim " +
            std::to_string(axis) + " (" + std::to_string(fiber->shape()[0]) +
            " vs " + std::to_string(tensor->shape()[axis]) + ")");
    }
    merge_axis(fiber->mutable_axes()[0], tensor->mutable_axes()[axis]);
    for(Index i = 0; i < batch_ndim; ++i)
    {
        Index ti = tensor->ndim() - batch_ndim + i;
        if(fiber->shape()[1 + i] != tensor->shape()[ti])
        {
            throw std::invalid_argument(
                op_name + ": fiber dim " + std::to_string(1 + i) +
                " must match tensor dim " + std::to_string(ti) + " (" +
                std::to_string(fiber->shape()[1 + i]) + " vs " +
                std::to_string(tensor->shape()[ti]) + ")");
        }
        merge_axis(fiber->mutable_axes()[1 + i],
                   tensor->mutable_axes()[ti]);
    }
}

//! Validate maxsumexp output shape and merge axes.
inline void validate_maxsumexp_shape_and_merge(
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst,
    Index axis,
    const std::string& op_name)
{
    if(dst->ndim() != src->ndim())
    {
        throw std::invalid_argument(
            op_name + ": dst ndim must equal src ndim (" +
            std::to_string(dst->ndim()) + " vs " +
            std::to_string(src->ndim()) + ")");
    }
    if(axis < 0 || axis >= src->ndim())
    {
        throw std::invalid_argument(op_name + ": axis out of range");
    }
    if(dst->shape()[0] != 2)
    {
        throw std::invalid_argument(
            op_name + ": dst dim 0 must be 2 (got " +
            std::to_string(dst->shape()[0]) + ")");
    }
    int d = 1;
    for(Index i = 0; i < src->ndim(); ++i)
    {
        if(i == axis)
            continue;
        if(dst->shape()[d] != src->shape()[i])
        {
            throw std::invalid_argument(
                op_name + ": shape mismatch at dimension " +
                std::to_string(d) + " (dst: " + std::to_string(dst->shape()[d]) +
                " vs src " + std::to_string(i) + ": " +
                std::to_string(src->shape()[i]) + ")");
        }
        merge_axis(src->mutable_axes()[i], dst->mutable_axes()[d]);
        ++d;
    }
}

//! Validate logsumexp output shape and merge axes.
inline void validate_logsumexp_shape_and_merge(
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst,
    const std::string& op_name)
{
    if(src->ndim() < 1 || dst->ndim() != src->ndim() - 1)
    {
        throw std::invalid_argument(
            op_name + ": dst ndim must equal src.ndim - 1 (" +
            std::to_string(dst->ndim()) + " vs " +
            std::to_string(src->ndim()) + ")");
    }
    for(Index i = 0; i < dst->ndim(); ++i)
    {
        if(dst->shape()[i] != src->shape()[i + 1])
        {
            throw std::invalid_argument(
                op_name + ": shape mismatch at dimension " + std::to_string(i) +
                " (dst: " + std::to_string(dst->shape()[i]) + " vs src: " +
                std::to_string(src->shape()[i + 1]) + ")");
        }
        merge_axis(src->mutable_axes()[i + 1], dst->mutable_axes()[i]);
    }
}

//! Validate embedding output shape and merge axes.
inline void validate_embedding_shape_and_merge(
    TensorGraph::TensorNode* embed,
    TensorGraph::TensorNode* index,
    TensorGraph::TensorNode* vocab,
    const std::string& op_name)
{
    if(embed->ndim() != index->ndim() + 1)
    {
        throw std::invalid_argument(
            op_name + ": embed ndim must equal index.ndim + 1 (" +
            std::to_string(embed->ndim()) + " vs " +
            std::to_string(index->ndim()) + ")");
    }
    for(Index i = 0; i < index->ndim(); ++i)
    {
        if(embed->shape()[i] != index->shape()[i])
        {
            throw std::invalid_argument(
                op_name + ": embed.dim[" + std::to_string(i) +
                "] must match index.dim[" + std::to_string(i) + "] (" +
                std::to_string(embed->shape()[i]) + " vs " +
                std::to_string(index->shape()[i]) + ")");
        }
        merge_axis(embed->mutable_axes()[i], index->mutable_axes()[i]);
    }
    if(embed->shape()[index->ndim()] != vocab->dim(0))
    {
        throw std::invalid_argument(
            op_name + ": embed.dim[" + std::to_string(index->ndim()) +
            "] must match vocab.dim[0] (" +
            std::to_string(embed->shape()[index->ndim()]) + " vs " +
            std::to_string(vocab->dim(0)) + ")");
    }
    merge_axis(embed->mutable_axes()[index->ndim()],
               vocab->mutable_axes()[0]);
}

} // namespace nntile::graph
