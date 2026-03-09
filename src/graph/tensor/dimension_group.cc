/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/dimension_group.cc
 * Dimension group discovery via union-find on tensor graph operations.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/dimension_group.hh"
#include "nntile/graph/tensor/graph.hh"
#include "nntile/graph/tensor/graph_ops.hh"

#include <algorithm>
#include <stdexcept>
#include <unordered_map>

namespace nntile::graph
{

namespace
{

// -----------------------------------------------------------------------
// Union-Find
// -----------------------------------------------------------------------
class UnionFind
{
public:
    void add(DimId id)
    {
        if(parent_.count(id) == 0)
        {
            parent_[id] = id;
            rank_[id] = 0;
        }
    }

    DimId find(DimId id)
    {
        DimId& p = parent_.at(id);
        if(!(p == id))
        {
            p = find(p);
        }
        return p;
    }

    void merge(DimId a, DimId b)
    {
        a = find(a);
        b = find(b);
        if(a == b) return;
        if(rank_[a] < rank_[b]) std::swap(a, b);
        parent_[b] = a;
        if(rank_[a] == rank_[b]) ++rank_[a];
    }

    std::map<DimId, std::vector<DimId>> groups() const
    {
        std::map<DimId, std::vector<DimId>> result;
        auto self = const_cast<UnionFind*>(this);
        for(auto& [id, _] : parent_)
        {
            DimId root = self->find(id);
            result[root].push_back(id);
        }
        return result;
    }

private:
    std::map<DimId, DimId> parent_;
    std::map<DimId, int> rank_;
};

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------
using TN = const TensorGraph::TensorNode*;

void merge_all_dims(std::vector<DimConstraint>& out, TN a, TN b)
{
    int ndim = static_cast<int>(a->shape().size());
    for(int i = 0; i < ndim; ++i)
    {
        out.push_back({{a, i}, {b, i}});
    }
}

// -----------------------------------------------------------------------
// Constraint extraction per operation type
// -----------------------------------------------------------------------

void constraints_elementwise_same_shape(
    std::vector<DimConstraint>& out,
    const std::vector<TN>& tensors)
{
    if(tensors.size() < 2) return;
    TN first = tensors[0];
    for(size_t t = 1; t < tensors.size(); ++t)
    {
        merge_all_dims(out, first, tensors[t]);
    }
}

void constraints_gemm(std::vector<DimConstraint>& out,
                      const tensor::TensorGemmOp& op)
{
    TN a = op.a;
    TN b = op.b;
    TN c = op.c;
    int a_ndim = static_cast<int>(a->shape().size());
    int b_ndim = static_cast<int>(b->shape().size());
    int ndim = static_cast<int>(op.ndim);
    int batch_ndim = static_cast<int>(op.batch_ndim);

    int a_batch_start = a_ndim - batch_ndim;
    int b_batch_start = b_ndim - batch_ndim;

    int a_m_begin, a_m_end, a_k_begin;
    if(!op.trans_a)
    {
        a_m_begin = 0;
        a_m_end = a_batch_start - ndim;
        a_k_begin = a_m_end;
    }
    else
    {
        a_k_begin = 0;
        a_m_begin = ndim;
        a_m_end = a_batch_start;
    }
    int num_m = a_m_end - a_m_begin;

    int b_k_begin, b_n_begin, b_n_end;
    if(!op.trans_b)
    {
        b_k_begin = 0;
        b_n_begin = ndim;
        b_n_end = b_batch_start;
    }
    else
    {
        b_n_begin = 0;
        b_n_end = b_batch_start - ndim;
        b_k_begin = b_n_end;
    }
    int num_n = b_n_end - b_n_begin;

    // C layout: [m_dims..., n_dims..., batch_dims...]
    // M dims: A's M dims == C's first num_m dims
    for(int i = 0; i < num_m; ++i)
    {
        out.push_back({{a, a_m_begin + i}, {c, i}});
    }

    // K dims: A's K dims == B's K dims (contraction)
    for(int i = 0; i < ndim; ++i)
    {
        out.push_back({{a, a_k_begin + i}, {b, b_k_begin + i}});
    }

    // N dims: B's N dims == C's dims [num_m..num_m+num_n)
    for(int i = 0; i < num_n; ++i)
    {
        out.push_back({{b, b_n_begin + i}, {c, num_m + i}});
    }

    // Batch dims: A, B, C all share trailing batch dims
    int c_batch_start = num_m + num_n;
    for(int i = 0; i < batch_ndim; ++i)
    {
        out.push_back({{a, a_batch_start + i}, {c, c_batch_start + i}});
        out.push_back({{b, b_batch_start + i}, {c, c_batch_start + i}});
    }
}

void constraints_fiber_op(
    std::vector<DimConstraint>& out,
    TN fiber, TN full_tensor, TN output,
    Index axis, Index batch_ndim)
{
    int t_ndim = static_cast<int>(full_tensor->shape().size());

    // fiber.dim[0] == full_tensor.dim[axis]
    out.push_back({{fiber, 0}, {full_tensor, static_cast<int>(axis)}});

    // fiber batch dims == full_tensor batch dims
    for(int i = 0; i < static_cast<int>(batch_ndim); ++i)
    {
        out.push_back({{fiber, 1 + i},
                        {full_tensor, t_ndim - static_cast<int>(batch_ndim) + i}});
    }

    // output (if different from full_tensor) has same shape as full_tensor
    if(output != nullptr && output != full_tensor)
    {
        merge_all_dims(out, full_tensor, output);
    }
}

void constraints_slice_reduce(
    std::vector<DimConstraint>& out,
    TN src, TN dst, Index axis)
{
    // src has N dims, dst has N-1 dims (axis removed)
    int src_ndim = static_cast<int>(src->shape().size());
    int d = 0;
    for(int s = 0; s < src_ndim; ++s)
    {
        if(s == static_cast<int>(axis)) continue;
        out.push_back({{src, s}, {dst, d}});
        ++d;
    }
}

void constraints_slice_broadcast(
    std::vector<DimConstraint>& out,
    TN slice, TN full_tensor, TN output, Index axis)
{
    // slice has N-1 dims (all of full_tensor except axis)
    int t_ndim = static_cast<int>(full_tensor->shape().size());
    int d = 0;
    for(int s = 0; s < t_ndim; ++s)
    {
        if(s == static_cast<int>(axis)) continue;
        out.push_back({{slice, d}, {full_tensor, s}});
        ++d;
    }

    if(output != nullptr && output != full_tensor)
    {
        merge_all_dims(out, full_tensor, output);
    }
}

void constraints_transpose(
    std::vector<DimConstraint>& out,
    TN src, TN dst, Index ndim)
{
    // output_shape[i] = src_shape[(i + ndim) % src.ndim()]
    int n = static_cast<int>(src->shape().size());
    int nd = static_cast<int>(ndim);
    for(int i = 0; i < n; ++i)
    {
        int src_dim = (i + nd) % n;
        out.push_back({{src, src_dim}, {dst, i}});
    }
}

} // namespace

// -----------------------------------------------------------------------
// get_tiling_constraints — dispatch by op_name
// -----------------------------------------------------------------------
std::vector<DimConstraint> get_tiling_constraints(
    const TensorGraph::OpNode& op)
{
    std::vector<DimConstraint> out;
    const std::string& name = op.op_name();

    // --- Elementwise same-shape ---
    if(name == "ADD")
    {
        auto& o = static_cast<const tensor::TensorAddOp&>(op);
        constraints_elementwise_same_shape(out, {o.x, o.y, o.z});
    }
    else if(name == "ADD_INPLACE")
    {
        auto& o = static_cast<const tensor::TensorAddInplaceOp&>(op);
        constraints_elementwise_same_shape(out, {o.x, o.y});
    }
    else if(name == "MULTIPLY")
    {
        auto& o = static_cast<const tensor::TensorMultiplyOp&>(op);
        constraints_elementwise_same_shape(out, {o.x, o.y, o.z});
    }
    else if(name == "MULTIPLY_INPLACE")
    {
        auto& o = static_cast<const tensor::TensorMultiplyInplaceOp&>(op);
        constraints_elementwise_same_shape(out, {o.src, o.dst});
    }
    else if(name == "HYPOT")
    {
        auto& o = static_cast<const tensor::TensorHypotOp&>(op);
        constraints_elementwise_same_shape(out, {o.src1, o.src2, o.dst});
    }
    else if(name == "HYPOT_INPLACE")
    {
        auto& o = static_cast<const tensor::TensorHypotInplaceOp&>(op);
        constraints_elementwise_same_shape(out, {o.src, o.dst});
    }
    else if(name == "GELU")
    {
        auto& o = static_cast<const tensor::TensorGeluOp&>(op);
        constraints_elementwise_same_shape(out, {o.x, o.y});
    }
    else if(name == "GELU_BACKWARD")
    {
        auto& o = static_cast<const tensor::TensorGeluBackwardOp&>(op);
        constraints_elementwise_same_shape(out, {o.x, o.dy, o.dx});
    }
    else if(name == "GELUTANH")
    {
        auto& o = static_cast<const tensor::TensorGelutanhOp&>(op);
        constraints_elementwise_same_shape(out, {o.src, o.dst});
    }
    else if(name == "GELUTANH_BACKWARD")
    {
        auto& o = static_cast<const tensor::TensorGelutanhBackwardOp&>(op);
        constraints_elementwise_same_shape(out, {o.x, o.dy, o.dx});
    }
    else if(name == "RELU")
    {
        auto& o = static_cast<const tensor::TensorReluOp&>(op);
        constraints_elementwise_same_shape(out, {o.src, o.dst});
    }
    else if(name == "RELU_BACKWARD")
    {
        auto& o = static_cast<const tensor::TensorReluBackwardOp&>(op);
        constraints_elementwise_same_shape(out, {o.x, o.dy, o.dx});
    }
    else if(name == "SILU")
    {
        auto& o = static_cast<const tensor::TensorSiluOp&>(op);
        constraints_elementwise_same_shape(out, {o.src, o.dst});
    }
    else if(name == "SILU_BACKWARD")
    {
        auto& o = static_cast<const tensor::TensorSiluBackwardOp&>(op);
        constraints_elementwise_same_shape(out, {o.x, o.dy, o.dx});
    }
    else if(name == "SCALE")
    {
        auto& o = static_cast<const tensor::TensorScaleOp&>(op);
        constraints_elementwise_same_shape(out, {o.src, o.dst});
    }
    else if(name == "COPY")
    {
        auto& o = static_cast<const tensor::TensorCopyOp&>(op);
        constraints_elementwise_same_shape(out, {o.src, o.dst});
    }
    else if(name == "SQRT")
    {
        auto& o = static_cast<const tensor::TensorSqrtOp&>(op);
        constraints_elementwise_same_shape(out, {o.src, o.dst});
    }
    else if(name == "SOFTMAX")
    {
        auto& o = static_cast<const tensor::TensorSoftmaxOp&>(op);
        constraints_elementwise_same_shape(out, {o.src, o.dst});
    }
    else if(name == "SOFTMAX_INPLACE")
    {
        // dst only (self)
    }
    else if(name == "NORM")
    {
        auto& o = static_cast<const tensor::TensorNormOp&>(op);
        constraints_elementwise_same_shape(out, {o.x, o.y});
    }
    else if(name == "SUM")
    {
        auto& o = static_cast<const tensor::TensorSumOp&>(op);
        constraints_elementwise_same_shape(out, {o.src, o.dst});
    }
    // --- Inplace-only ops (single tensor, no cross-tensor constraints) ---
    else if(name == "GELU_INPLACE" || name == "GELUTANH_INPLACE" ||
            name == "RELU_INPLACE" || name == "SILU_INPLACE" ||
            name == "SCALE_INPLACE" || name == "SQRT_INPLACE" ||
            name == "HYPOT_SCALAR_INVERSE" || name == "POW" ||
            name == "RANDN" || name == "FILL" || name == "CLEAR")
    {
        // No cross-tensor constraints
    }
    // --- Optimizer steps (all same-shape tensors) ---
    else if(name == "SGD_STEP")
    {
        auto& o = static_cast<const tensor::TensorSgdStepOp&>(op);
        constraints_elementwise_same_shape(out, {o.grad, o.velocity, o.p});
    }
    else if(name == "ADAM_STEP")
    {
        auto& o = static_cast<const tensor::TensorAdamStepOp&>(op);
        constraints_elementwise_same_shape(
            out, {o.grad, o.first_moment, o.second_moment, o.p});
    }
    else if(name == "ADAMW_STEP")
    {
        auto& o = static_cast<const tensor::TensorAdamwStepOp&>(op);
        constraints_elementwise_same_shape(
            out, {o.grad, o.first_moment, o.second_moment, o.p});
    }
    // --- GEMM ---
    else if(name == "GEMM")
    {
        auto& o = static_cast<const tensor::TensorGemmOp&>(op);
        constraints_gemm(out, o);
    }
    // --- Fiber operations ---
    else if(name == "ADD_FIBER")
    {
        auto& o = static_cast<const tensor::TensorAddFiberOp&>(op);
        constraints_fiber_op(out, o.fiber, o.tensor, o.output,
                             o.axis, o.batch_ndim);
    }
    else if(name == "ADD_FIBER_INPLACE")
    {
        auto& o = static_cast<const tensor::TensorAddFiberInplaceOp&>(op);
        constraints_fiber_op(out, o.fiber, o.tensor, nullptr,
                             o.axis, o.batch_ndim);
    }
    else if(name == "SCALE_FIBER")
    {
        auto& o = static_cast<const tensor::TensorScaleFiberOp&>(op);
        // scale_fiber: src is fiber, dst is full tensor
        // fiber shape: [axis_extent, batch_dims...]
        constraints_fiber_op(out, o.src, o.dst, nullptr,
                             o.axis, o.batch_ndim);
    }
    else if(name == "MULTIPLY_FIBER")
    {
        auto& o = static_cast<const tensor::TensorMultiplyFiberOp&>(op);
        // src1 is fiber, src2 is full tensor, dst same shape as src2
        constraints_fiber_op(out, o.src1, o.src2, o.dst, o.axis, 0);
    }
    else if(name == "MULTIPLY_FIBER_INPLACE")
    {
        auto& o = static_cast<const tensor::TensorMultiplyFiberInplaceOp&>(op);
        constraints_fiber_op(out, o.src, o.dst, nullptr, o.axis, 0);
    }
    else if(name == "SUM_FIBER")
    {
        auto& o = static_cast<const tensor::TensorSumFiberOp&>(op);
        // x is full tensor, y is fiber. Reverse of add_fiber direction.
        constraints_fiber_op(out, o.y, o.x, nullptr,
                             o.axis, o.batch_ndim);
    }
    else if(name == "NORM_FIBER")
    {
        auto& o = static_cast<const tensor::TensorNormFiberOp&>(op);
        // src1 is full, dst is fiber, src2 is fiber (accumulator)
        constraints_fiber_op(out, o.dst, o.src1, nullptr,
                             o.axis, o.batch_ndim);
        constraints_elementwise_same_shape(out, {o.src2, o.dst});
    }
    else if(name == "NORM_FIBER_INPLACE")
    {
        auto& o = static_cast<const tensor::TensorNormFiberInplaceOp&>(op);
        constraints_fiber_op(out, o.dst, o.src, nullptr,
                             o.axis, o.batch_ndim);
    }
    else if(name == "SUMPROD_FIBER")
    {
        auto& o = static_cast<const tensor::TensorSumprodFiberOp&>(op);
        // src1 and src2 are full tensors, dst is fiber
        constraints_elementwise_same_shape(out, {o.src1, o.src2});
        constraints_fiber_op(out, o.dst, o.src1, nullptr, o.axis, 0);
    }
    // --- Slice operations ---
    else if(name == "ADD_SLICE")
    {
        auto& o = static_cast<const tensor::TensorAddSliceOp&>(op);
        // src1 is slice (N-1 dims), src2 is full (N dims), dst same as src2
        constraints_slice_broadcast(out, o.src1, o.src2, o.dst, o.axis);
    }
    else if(name == "ADD_SLICE_INPLACE")
    {
        auto& o = static_cast<const tensor::TensorAddSliceInplaceOp&>(op);
        // src is slice (N-1 dims), dst is full (N dims)
        constraints_slice_broadcast(out, o.src, o.dst, nullptr, o.axis);
    }
    else if(name == "SUM_SLICE")
    {
        auto& o = static_cast<const tensor::TensorSumSliceOp&>(op);
        // src is full (N dims), dst is slice (N-1 dims, axis removed)
        constraints_slice_reduce(out, o.src, o.dst, o.axis);
    }
    else if(name == "SCALE_SLICE")
    {
        auto& o = static_cast<const tensor::TensorScaleSliceOp&>(op);
        // src is slice (fiber along non-axis dims), dst is full tensor
        constraints_slice_broadcast(out, o.src, o.dst, nullptr, o.axis);
    }
    else if(name == "MULTIPLY_SLICE")
    {
        auto& o = static_cast<const tensor::TensorMultiplySliceOp&>(op);
        constraints_slice_broadcast(out, o.src, o.dst, nullptr, o.axis);
    }
    else if(name == "NORM_SLICE")
    {
        auto& o = static_cast<const tensor::TensorNormSliceOp&>(op);
        // src1 is full, dst is slice, src2 is slice (accumulator)
        constraints_slice_reduce(out, o.src1, o.dst, o.axis);
        constraints_elementwise_same_shape(out, {o.src2, o.dst});
    }
    else if(name == "NORM_SLICE_INPLACE")
    {
        auto& o = static_cast<const tensor::TensorNormSliceInplaceOp&>(op);
        constraints_slice_reduce(out, o.src, o.dst, o.axis);
    }
    else if(name == "SUMPROD_SLICE")
    {
        auto& o = static_cast<const tensor::TensorSumprodSliceOp&>(op);
        constraints_elementwise_same_shape(out, {o.src1, o.src2});
        constraints_slice_reduce(out, o.src1, o.dst, o.axis);
    }
    // --- Maxsumexp / Logsumexp (reduce one axis) ---
    else if(name == "MAXSUMEXP")
    {
        auto& o = static_cast<const tensor::TensorMaxsumexpOp&>(op);
        // src has N dims, dst has N dims but dst.dim[axis] == 2
        // (stores max and sumexp). Non-axis dims match.
        int n = static_cast<int>(o.src->shape().size());
        int d = 0;
        for(int s = 0; s < n; ++s)
        {
            if(s == static_cast<int>(o.axis))
            {
                ++d;
                continue;
            }
            out.push_back({{o.src, s}, {o.dst, d}});
            ++d;
        }
    }
    else if(name == "LOGSUMEXP")
    {
        auto& o = static_cast<const tensor::TensorLogsumexpOp&>(op);
        constraints_elementwise_same_shape(out, {o.src, o.dst});
    }
    // --- Transpose ---
    else if(name == "TRANSPOSE")
    {
        auto& o = static_cast<const tensor::TensorTransposeOp&>(op);
        constraints_transpose(out, o.src, o.dst, o.ndim);
    }
    // --- Rope ---
    else if(name == "ROPE")
    {
        auto& o = static_cast<const tensor::TensorRopeOp&>(op);
        constraints_elementwise_same_shape(out, {o.src, o.dst});
    }
    else if(name == "ROPE_BACKWARD")
    {
        auto& o = static_cast<const tensor::TensorRopeBackwardOp&>(op);
        constraints_elementwise_same_shape(out, {o.dy, o.dx});
    }
    // --- Mask scalar ---
    else if(name == "MASK_SCALAR")
    {
        // mask and A have related shapes but the relationship is complex
        // (batch_ndim-dependent). Leave as no constraints for now.
    }
    // --- Subtract indexed outputs ---
    else if(name == "SUBTRACT_INDEXED_OUTPUTS")
    {
        // labels and dst have complex shape relationship, skip
    }
    // --- Total sum accum ---
    else if(name == "TOTAL_SUM_ACCUM")
    {
        // Complex multi-tensor relationship, skip
    }
    // --- Embedding ---
    else if(name == "EMBEDDING" || name == "EMBEDDING_BACKWARD")
    {
        // Complex axis-dependent relationship, skip
    }
    // --- Conv2d ---
    else if(name == "CONV2D_INPLACE" || name == "CONV2D_BWD_INPUT_INPLACE" ||
            name == "CONV2D_BWD_WEIGHT_INPLACE")
    {
        // Complex spatial dimension relationships, skip
    }
    // --- Flash attention ---
    else if(name == "FLASH_SDPA_FWD_CUDNN" || name == "FLASH_SDPA_BWD_CUDNN")
    {
        // Complex multi-tensor attention relationships, skip
    }
    // --- Gather/Scatter ---
    else if(name == "GATHER" || name == "SCATTER")
    {
        // Redistribution ops, skip
    }
    // --- Copy intersection ---
    else if(name == "COPY_INTERSECTION")
    {
        // Partial copy with offsets, skip
    }
    // --- Log scalar (no output) ---
    else if(name == "LOG_SCALAR")
    {
        // No output tensor
    }

    return out;
}

// -----------------------------------------------------------------------
// discover_dimension_groups
// -----------------------------------------------------------------------
std::vector<DimensionGroup> discover_dimension_groups(
    const TensorGraph& graph)
{
    UnionFind uf;

    // Register all (tensor, dim) pairs
    for(const auto& node : graph.tensor_nodes())
    {
        int ndim = static_cast<int>(node->shape().size());
        for(int i = 0; i < ndim; ++i)
        {
            uf.add({node.get(), i});
        }
    }

    // Walk all operations and merge constrained dimensions
    for(const auto& op : graph.ops())
    {
        auto constraints = get_tiling_constraints(*op);
        for(const auto& c : constraints)
        {
            uf.merge(c.a, c.b);
        }
    }

    // Collect groups
    auto raw_groups = uf.groups();
    std::vector<DimensionGroup> result;
    result.reserve(raw_groups.size());

    for(auto& [root, members] : raw_groups)
    {
        DimensionGroup g;
        g.members = std::move(members);
        // Sort by (tensor name, axis) for stable output
        std::sort(g.members.begin(), g.members.end(),
            [](const DimId& a, const DimId& b)
            {
                int cmp = a.node->name().compare(b.node->name());
                if(cmp != 0) return cmp < 0;
                return a.axis < b.axis;
            });

        g.extent = g.members[0].node->shape()[
            static_cast<size_t>(g.members[0].axis)];

        // Auto-generate name from first member
        g.name = g.members[0].node->name() + ".dim" +
                 std::to_string(g.members[0].axis);

        result.push_back(std::move(g));
    }

    // Sort groups by name for deterministic order
    std::sort(result.begin(), result.end(),
        [](const DimensionGroup& a, const DimensionGroup& b)
        {
            return a.name < b.name;
        });

    return result;
}

} // namespace nntile::graph
