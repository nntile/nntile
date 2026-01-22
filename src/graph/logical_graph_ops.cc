/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/logical_graph_ops.cc
 * Logical graph operations.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/logical_graph_ops.hh"

// Include standard headers
#include <stdexcept>
#include <utility>

namespace nntile::graph
{

namespace
{

//! Compute output shape for gemm (tensor contraction) operation
//!
//! Tensor layout (column-major, dimensions listed from inner to outer):
//!
//! A (no trans): [M_dims..., K_dims..., batch_dims...]
//! A (trans):    [K_dims..., M_dims..., batch_dims...]
//!
//! B (no trans): [K_dims..., N_dims..., batch_dims...]
//! B (trans):    [N_dims..., K_dims..., batch_dims...]
//!
//! C:            [M_dims..., N_dims..., batch_dims...]
//!
//! Where:
//!   - ndim = number of contraction dimensions (K)
//!   - batch_ndim = number of batch dimensions (must match between A and B)
//!
std::vector<Index> compute_gemm_output_shape(
    const LogicalGraph::TensorNode& a,
    const LogicalGraph::TensorNode& b,
    bool trans_a,
    bool trans_b,
    Index ndim,
    Index batch_ndim)
{
    // Validate ndim and batch_ndim are non-negative
    if(ndim < 1)
    {
        throw std::invalid_argument(
            "gemm: ndim must be at least 1");
    }
    if(batch_ndim < 0)
    {
        throw std::invalid_argument(
            "gemm: batch_ndim must be non-negative");
    }

    // Validate tensor dimensions are sufficient
    // A needs at least ndim + batch_ndim dimensions (M can be empty for vector)
    // But typically A needs ndim + batch_ndim + (M_ndim >= 1)
    if(a.ndim() < ndim + batch_ndim)
    {
        throw std::invalid_argument(
            "gemm: tensor A has insufficient dimensions for given ndim and batch_ndim");
    }
    if(b.ndim() < ndim + batch_ndim)
    {
        throw std::invalid_argument(
            "gemm: tensor B has insufficient dimensions for given ndim and batch_ndim");
    }

    // Compute dimension counts
    Index a_m_ndim = a.ndim() - ndim - batch_ndim;  // Number of M dimensions
    Index b_n_ndim = b.ndim() - ndim - batch_ndim;  // Number of N dimensions

    // Extract shapes based on transpose flags
    // A: [M..., K..., batch...] or [K..., M..., batch...] if trans_a
    // B: [K..., N..., batch...] or [N..., K..., batch...] if trans_b

    // Get K dimensions from A
    std::vector<Index> k_dims_a(ndim);
    Index k_start_a = trans_a ? 0 : a_m_ndim;
    for(Index i = 0; i < ndim; ++i)
    {
        k_dims_a[i] = a.dim(k_start_a + i);
    }

    // Get K dimensions from B
    std::vector<Index> k_dims_b(ndim);
    Index k_start_b = trans_b ? b_n_ndim : 0;
    for(Index i = 0; i < ndim; ++i)
    {
        k_dims_b[i] = b.dim(k_start_b + i);
    }

    // Validate K dimensions match
    for(Index i = 0; i < ndim; ++i)
    {
        if(k_dims_a[i] != k_dims_b[i])
        {
            throw std::invalid_argument(
                "gemm: contraction dimension " + std::to_string(i) +
                " mismatch: A has " + std::to_string(k_dims_a[i]) +
                ", B has " + std::to_string(k_dims_b[i]));
        }
    }

    // Get batch dimensions from A and B
    std::vector<Index> batch_dims_a(batch_ndim);
    std::vector<Index> batch_dims_b(batch_ndim);
    for(Index i = 0; i < batch_ndim; ++i)
    {
        batch_dims_a[i] = a.dim(a.ndim() - batch_ndim + i);
        batch_dims_b[i] = b.dim(b.ndim() - batch_ndim + i);
    }

    // Validate batch dimensions match
    for(Index i = 0; i < batch_ndim; ++i)
    {
        if(batch_dims_a[i] != batch_dims_b[i])
        {
            throw std::invalid_argument(
                "gemm: batch dimension " + std::to_string(i) +
                " mismatch: A has " + std::to_string(batch_dims_a[i]) +
                ", B has " + std::to_string(batch_dims_b[i]));
        }
    }

    // Build output shape: [M..., N..., batch...]
    std::vector<Index> output_shape;
    output_shape.reserve(a_m_ndim + b_n_ndim + batch_ndim);

    // Add M dimensions from A
    Index m_start_a = trans_a ? ndim : 0;
    for(Index i = 0; i < a_m_ndim; ++i)
    {
        output_shape.push_back(a.dim(m_start_a + i));
    }

    // Add N dimensions from B
    Index n_start_b = trans_b ? 0 : ndim;
    for(Index i = 0; i < b_n_ndim; ++i)
    {
        output_shape.push_back(b.dim(n_start_b + i));
    }

    // Add batch dimensions
    for(Index i = 0; i < batch_ndim; ++i)
    {
        output_shape.push_back(batch_dims_a[i]);
    }

    return output_shape;
}

} // namespace

//! Clear tensor: x = 0
void clear(LogicalGraph::TensorNode& x)
{
    OpAttrs attrs = ClearAttrs{};

    // In-place operation: inputs and outputs are the same tensor
    x.graph().add_op(
        OpType::CLEAR,
        attrs,
        {},
        {&x}
    );
}

//! GeLU activation: y = gelu(x)
LogicalGraph::TensorNode& gelu(
    LogicalGraph::TensorNode& x,
    const std::string& output_name)
{
    // Output shape = input shape
    std::vector<Index> output_shape = x.shape();

    // Create output tensor
    LogicalGraph::TensorNode& output = x.graph().tensor(
        std::move(output_shape),
        output_name,
        x.dtype());

    // Create operation attributes
    OpAttrs attrs = GeluAttrs{};

    // Add operation to graph using public builder API
    x.graph().add_op(
        OpType::GELU,
        attrs,
        {&x},
        {&output}
    );

    return output;
}

//! GeLU backward: dx += gelu_backward(x, dy)
void gelu_backward(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& dy,
    LogicalGraph::TensorNode& dx)
{
    // Create operation attributes
    OpAttrs attrs = GeluBackwardAttrs{};

    // Add operation to graph using public builder API
    // Note: dx is both input and output (accumulates gradients)
    x.graph().add_op(
        OpType::GELU_BACKWARD,
        attrs,
        {&x, &dy, &dx},
        {&dx}
    );
}

//! GeLU in-place: x = gelu(x)
void gelu_inplace(LogicalGraph::TensorNode& x)
{
    OpAttrs attrs = GeluAttrs{};
    x.graph().add_op(
        OpType::GELU_INPLACE,
        attrs,
        {&x},
        {&x}
    );
}

//! GeLU tanh activation: y = gelu_tanh(x)
LogicalGraph::TensorNode& gelutanh(
    LogicalGraph::TensorNode& x,
    const std::string& output_name)
{
    // Output shape = input shape
    std::vector<Index> output_shape = x.shape();

    // Create output tensor
    LogicalGraph::TensorNode& output = x.graph().tensor(
        std::move(output_shape),
        output_name,
        x.dtype());

    // Create operation attributes
    OpAttrs attrs = GeluAttrs{};  // Reuse GeluAttrs for now

    // Add operation to graph
    x.graph().add_op(
        OpType::GELUTANH,
        attrs,
        {&x},
        {&output}
    );

    return output;
}

//! GeLU tanh in-place: x = gelu_tanh(x)
void gelutanh_inplace(LogicalGraph::TensorNode& x)
{
    OpAttrs attrs = GeluAttrs{};
    x.graph().add_op(
        OpType::GELUTANH_INPLACE,
        attrs,
        {&x},
        {&x}
    );
}

//! GeLU tanh backward: dx += gelu_tanh_backward(x, dy)
void gelutanh_backward(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& dy,
    LogicalGraph::TensorNode& dx)
{
    OpAttrs attrs = GeluBackwardAttrs{};
    x.graph().add_op(
        OpType::GELUTANH_BACKWARD,
        attrs,
        {&x, &dy, &dx},
        {&dx}
    );
}

//! ReLU activation: y = relu(x)
LogicalGraph::TensorNode& relu(
    LogicalGraph::TensorNode& x,
    const std::string& output_name)
{
    std::vector<Index> output_shape = x.shape();
    LogicalGraph::TensorNode& output = x.graph().tensor(
        std::move(output_shape),
        output_name,
        x.dtype());

    OpAttrs attrs = GeluAttrs{};  // Reuse for now
    x.graph().add_op(
        OpType::RELU,
        attrs,
        {&x},
        {&output}
    );

    return output;
}

//! ReLU in-place: x = relu(x)
void relu_inplace(LogicalGraph::TensorNode& x)
{
    OpAttrs attrs = GeluAttrs{};
    x.graph().add_op(
        OpType::RELU_INPLACE,
        attrs,
        {&x},
        {&x}
    );
}

//! ReLU backward: dx += relu_backward(x, dy)
void relu_backward(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& dy,
    LogicalGraph::TensorNode& dx)
{
    OpAttrs attrs = GeluBackwardAttrs{};
    x.graph().add_op(
        OpType::RELU_BACKWARD,
        attrs,
        {&x, &dy, &dx},
        {&dx}
    );
}

//! SiLU activation: y = silu(x)
LogicalGraph::TensorNode& silu(
    LogicalGraph::TensorNode& x,
    const std::string& output_name)
{
    std::vector<Index> output_shape = x.shape();
    LogicalGraph::TensorNode& output = x.graph().tensor(
        std::move(output_shape),
        output_name,
        x.dtype());

    OpAttrs attrs = GeluAttrs{};
    x.graph().add_op(
        OpType::SILU,
        attrs,
        {&x},
        {&output}
    );

    return output;
}

//! SiLU in-place: x = silu(x)
void silu_inplace(LogicalGraph::TensorNode& x)
{
    OpAttrs attrs = GeluAttrs{};
    x.graph().add_op(
        OpType::SILU_INPLACE,
        attrs,
        {&x},
        {&x}
    );
}

//! SiLU backward: dx += silu_backward(x, dy)
void silu_backward(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& dy,
    LogicalGraph::TensorNode& dx)
{
    OpAttrs attrs = GeluBackwardAttrs{};
    x.graph().add_op(
        OpType::SILU_BACKWARD,
        attrs,
        {&x, &dy, &dx},
        {&dx}
    );
}

//! Sqrt activation: y = sqrt(x)
LogicalGraph::TensorNode& sqrt(
    LogicalGraph::TensorNode& x,
    const std::string& output_name)
{
    std::vector<Index> output_shape = x.shape();
    LogicalGraph::TensorNode& output = x.graph().tensor(
        std::move(output_shape),
        output_name,
        x.dtype());

    OpAttrs attrs = GeluAttrs{};
    x.graph().add_op(
        OpType::SQRT,
        attrs,
        {&x},
        {&output}
    );

    return output;
}

//! Sqrt in-place: x = sqrt(x)
void sqrt_inplace(LogicalGraph::TensorNode& x)
{
    OpAttrs attrs = GeluAttrs{};
    x.graph().add_op(
        OpType::SQRT_INPLACE,
        attrs,
        {&x},
        {&x}
    );
}

//! Validate inputs for binary operations
void validate_binary_inputs(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& y,
    LogicalGraph& expected_graph)
{
    if(&x.graph() != &expected_graph || &y.graph() != &expected_graph)
    {
        throw std::invalid_argument(
            "Binary operation: input tensors must belong to the same graph");
    }

    if(x.dtype() != y.dtype())
    {
        throw std::invalid_argument(
            "Binary operation: input tensors must have the same dtype");
    }

    if(x.shape() != y.shape())
    {
        throw std::invalid_argument(
            "Binary operation: input tensors must have the same shape");
    }
}

//! Add operation: z = alpha * x + beta * y
LogicalGraph::TensorNode& add(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& y,
    const std::string& output_name,
    Scalar alpha,
    Scalar beta)
{
    validate_binary_inputs(x, y, x.graph());

    std::vector<Index> output_shape = x.shape();
    LogicalGraph::TensorNode& output = x.graph().tensor(
        std::move(output_shape),
        output_name,
        x.dtype());

    OpAttrs attrs = BinaryOpAttrs{alpha, beta};
    x.graph().add_op(
        OpType::ADD,
        attrs,
        {&x, &y},
        {&output}
    );

    return output;
}

//! Add in-place: y = alpha * x + beta * y
void add_inplace(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& y,
    Scalar alpha,
    Scalar beta)
{
    validate_binary_inputs(x, y, x.graph());

    OpAttrs attrs = BinaryOpAttrs{alpha, beta};
    x.graph().add_op(
        OpType::ADD_INPLACE,
        attrs,
        {&x, &y},
        {&y}
    );
}

//! Multiply operation: z = x * y
LogicalGraph::TensorNode& multiply(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& y,
    const std::string& output_name)
{
    validate_binary_inputs(x, y, x.graph());

    std::vector<Index> output_shape = x.shape();
    LogicalGraph::TensorNode& output = x.graph().tensor(
        std::move(output_shape),
        output_name,
        x.dtype());

    OpAttrs attrs = BinaryOpAttrs{1.0, 1.0};  // alpha=1, beta=1 for multiply
    x.graph().add_op(
        OpType::MULTIPLY,
        attrs,
        {&x, &y},
        {&output}
    );

    return output;
}

//! Multiply in-place: y = x * y
void multiply_inplace(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& y)
{
    validate_binary_inputs(x, y, x.graph());

    OpAttrs attrs = BinaryOpAttrs{1.0, 1.0};
    x.graph().add_op(
        OpType::MULTIPLY_INPLACE,
        attrs,
        {&x, &y},
        {&y}
    );
}

//! Hypot operation: z = hypot(alpha * x, beta * y)
LogicalGraph::TensorNode& hypot(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& y,
    const std::string& output_name,
    Scalar alpha,
    Scalar beta)
{
    validate_binary_inputs(x, y, x.graph());

    std::vector<Index> output_shape = x.shape();
    LogicalGraph::TensorNode& output = x.graph().tensor(
        std::move(output_shape),
        output_name,
        x.dtype());

    OpAttrs attrs = BinaryOpAttrs{alpha, beta};
    x.graph().add_op(
        OpType::HYPOT,
        attrs,
        {&x, &y},
        {&output}
    );

    return output;
}

//! Hypot in-place: y = hypot(alpha * x, beta * y)
void hypot_inplace(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& y,
    Scalar alpha,
    Scalar beta)
{
    validate_binary_inputs(x, y, x.graph());

    OpAttrs attrs = BinaryOpAttrs{alpha, beta};
    x.graph().add_op(
        OpType::HYPOT_INPLACE,
        attrs,
        {&x, &y},
        {&y}
    );
}

//! Power operation: y = alpha * (x ^ exp)
LogicalGraph::TensorNode& pow(
    LogicalGraph::TensorNode& x,
    const std::string& output_name,
    Scalar alpha,
    Scalar exp)
{
    std::vector<Index> output_shape = x.shape();
    LogicalGraph::TensorNode& output = x.graph().tensor(
        std::move(output_shape),
        output_name,
        x.dtype());

    OpAttrs attrs = PowAttrs{alpha, exp};
    x.graph().add_op(
        OpType::POW,
        attrs,
        {&x},
        {&output}
    );

    return output;
}

//! Power in-place: x = alpha * (x ^ exp)
void pow_inplace(
    LogicalGraph::TensorNode& x,
    Scalar alpha,
    Scalar exp)
{
    OpAttrs attrs = PowAttrs{alpha, exp};
    x.graph().add_op(
        OpType::POW_INPLACE,
        attrs,
        {&x},
        {&x}
    );
}

//! Log scalar operation: log value with given name
void log_scalar(
    LogicalGraph::TensorNode& x,
    const std::string& name)
{
    OpAttrs attrs = LogScalarAttrs{name};
    x.graph().add_op(
        OpType::LOG_SCALAR,
        attrs,
        {&x},
        {}
    );
}

//! Mask scalar operation: conditionally set values based on mask
void mask_scalar(
    LogicalGraph::TensorNode& mask,
    LogicalGraph::TensorNode& x,
    Scalar val,
    Index batch_ndim)
{
    if(&mask.graph() != &x.graph())
    {
        throw std::invalid_argument(
            "mask_scalar: tensors must belong to the same graph");
    }

    if(mask.dtype() != DataType::BOOL)
    {
        throw std::invalid_argument(
            "mask_scalar: mask tensor must have bool dtype");
    }

    OpAttrs attrs = MaskScalarAttrs{val, batch_ndim};
    mask.graph().add_op(
        OpType::MASK_SCALAR,
        attrs,
        {&mask, &x},
        {&x}
    );
}

//! Fill operation: x = val
void fill(
    LogicalGraph::TensorNode& x,
    Scalar val)
{
    OpAttrs attrs = FillAttrs{val};
    x.graph().add_op(
        OpType::FILL,
        attrs,
        {},
        {&x}
    );
}

//! Copy operation: y = x
LogicalGraph::TensorNode& copy(
    LogicalGraph::TensorNode& x,
    const std::string& output_name)
{
    std::vector<Index> output_shape = x.shape();
    LogicalGraph::TensorNode& output = x.graph().tensor(
        std::move(output_shape),
        output_name,
        x.dtype());

    OpAttrs attrs = ClearAttrs{};  // No attributes needed
    x.graph().add_op(
        OpType::COPY,
        attrs,
        {&x},
        {&output}
    );

    return output;
}

//! Transpose operation: y = alpha * transpose(x)
LogicalGraph::TensorNode& transpose(
    LogicalGraph::TensorNode& x,
    const std::string& output_name,
    Scalar alpha,
    Index ndim)
{
    // For transpose, we need to reverse the first ndim dimensions
    std::vector<Index> output_shape = x.shape();
    if(ndim > 0 && ndim <= x.ndim())
    {
        // Reverse the first ndim dimensions
        for(Index i = 0; i < ndim/2; ++i)
        {
            std::swap(output_shape[i], output_shape[ndim-1-i]);
        }
    }

    LogicalGraph::TensorNode& output = x.graph().tensor(
        std::move(output_shape),
        output_name,
        x.dtype());

    OpAttrs attrs = TransposeAttrs{alpha, ndim};
    x.graph().add_op(
        OpType::TRANSPOSE,
        attrs,
        {&x},
        {&output}
    );

    return output;
}

//! Gather operation: y = gather(x)
LogicalGraph::TensorNode& gather(
    LogicalGraph::TensorNode& x,
    const std::string& output_name)
{
    // For now, assume gather doesn't change shape
    // In practice, this would depend on the indices
    std::vector<Index> output_shape = x.shape();
    LogicalGraph::TensorNode& output = x.graph().tensor(
        std::move(output_shape),
        output_name,
        x.dtype());

    OpAttrs attrs = ClearAttrs{};  // No attributes needed
    x.graph().add_op(
        OpType::GATHER,
        attrs,
        {&x},
        {&output}
    );

    return output;
}

//! Hypot scalar inverse operation: y = 1.0 / hypot(eps, alpha * y)
void hypot_scalar_inverse(
    LogicalGraph::TensorNode& x,
    Scalar eps,
    Scalar alpha)
{
    OpAttrs attrs = HypotScalarInverseAttrs{eps, alpha};
    x.graph().add_op(
        OpType::HYPOT_SCALAR_INVERSE,
        attrs,
        {&x},
        {&x}
    );
}

//! Subtract indexed outputs operation: subtract val from elements indexed by labels
void subtract_indexed_outputs(
    LogicalGraph::TensorNode& labels,
    LogicalGraph::TensorNode& x,
    Scalar val,
    Index ignore_index)
{
    if(&labels.graph() != &x.graph())
    {
        throw std::invalid_argument(
            "subtract_indexed_outputs: tensors must belong to the same graph");
    }

    if(labels.dtype() != DataType::INT64)
    {
        throw std::invalid_argument(
            "subtract_indexed_outputs: labels tensor must have int64 dtype");
    }

    OpAttrs attrs = SubtractIndexedOutputsAttrs{val, ignore_index};
    labels.graph().add_op(
        OpType::SUBTRACT_INDEXED_OUTPUTS,
        attrs,
        {&labels, &x},
        {&x}
    );
}

//! Scatter operation: y = scatter(x)
LogicalGraph::TensorNode& scatter(
    LogicalGraph::TensorNode& x,
    const std::string& output_name)
{
    // For now, assume scatter doesn't change shape
    // In practice, this would depend on the indices
    std::vector<Index> output_shape = x.shape();
    LogicalGraph::TensorNode& output = x.graph().tensor(
        std::move(output_shape),
        output_name,
        x.dtype());

    OpAttrs attrs = ClearAttrs{};  // No attributes needed
    x.graph().add_op(
        OpType::SCATTER,
        attrs,
        {&x},
        {&output}
    );

    return output;
}

//! Copy intersection operation: copy overlapping regions between tensors
void copy_intersection(
    LogicalGraph::TensorNode& src,
    const std::vector<Index>& src_offset,
    LogicalGraph::TensorNode& dst,
    const std::vector<Index>& dst_offset)
{
    if(&src.graph() != &dst.graph())
    {
        throw std::invalid_argument(
            "copy_intersection: tensors must belong to the same graph");
    }

    if(src.dtype() != dst.dtype())
    {
        throw std::invalid_argument(
            "copy_intersection: tensors must have the same dtype");
    }

    OpAttrs attrs = CopyIntersectionAttrs{src_offset, dst_offset};
    src.graph().add_op(
        OpType::COPY_INTERSECTION,
        attrs,
        {&src, &dst},
        {&dst}
    );
}

//! Scale along fibers: y = alpha * scale_fiber(x, y)
void scale_fiber(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& y,
    Scalar alpha,
    Index axis,
    Index batch_ndim)
{
    if(&x.graph() != &y.graph())
    {
        throw std::invalid_argument(
            "scale_fiber: tensors must belong to the same graph");
    }

    if(x.dtype() != y.dtype())
    {
        throw std::invalid_argument(
            "scale_fiber: tensors must have the same dtype");
    }

    if(axis < 0 || axis >= x.ndim())
    {
        throw std::invalid_argument(
            "scale_fiber: axis out of bounds");
    }

    if(batch_ndim < 0 || axis + batch_ndim > x.ndim())
    {
        throw std::invalid_argument(
            "scale_fiber: invalid batch_ndim");
    }

    OpAttrs attrs = ReductionAttrs{alpha, 0.0, axis, batch_ndim, 0};  // beta=0, redux=0
    x.graph().add_op(
        OpType::SCALE_FIBER,
        attrs,
        {&x, &y},
        {&y}
    );
}

//! Scale along slices: y = alpha * scale_slice(x, y)
void scale_slice(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& y,
    Scalar alpha,
    Index axis)
{
    if(&x.graph() != &y.graph())
    {
        throw std::invalid_argument(
            "scale_slice: tensors must belong to the same graph");
    }

    if(x.dtype() != y.dtype())
    {
        throw std::invalid_argument(
            "scale_slice: tensors must have the same dtype");
    }

    if(axis < 0 || axis >= x.ndim())
    {
        throw std::invalid_argument(
            "scale_slice: axis out of bounds");
    }

    OpAttrs attrs = ReductionAttrs{alpha, 0.0, axis, 0, 0};  // batch_ndim=0, redux=0
    x.graph().add_op(
        OpType::SCALE_SLICE,
        attrs,
        {&x, &y},
        {&y}
    );
}

//! Total sum of all elements: y = alpha * sum(x) + beta * y
void sum(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& y,
    Scalar alpha,
    Scalar beta)
{
    if(&x.graph() != &y.graph())
    {
        throw std::invalid_argument(
            "sum: tensors must belong to the same graph");
    }

    if(y.ndim() != 1 || y.shape()[0] != 1)
    {
        throw std::invalid_argument(
            "sum: output tensor must be scalar (shape [1])");
    }

    if(x.dtype() != y.dtype())
    {
        throw std::invalid_argument(
            "sum: input and output tensors must have the same dtype");
    }

    OpAttrs attrs = TotalSumAttrs{alpha, beta};
    x.graph().add_op(
        OpType::SUM,
        attrs,
        {&x, &y},
        {&y}
    );
}

//! Sum along fibers: y = alpha * sum_fiber(x) + beta * y
void sum_fiber(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& y,
    Index axis,
    Index batch_ndim,
    int redux,
    Scalar alpha,
    Scalar beta)
{
    if(&x.graph() != &y.graph())
    {
        throw std::invalid_argument(
            "sum_fiber: tensors must belong to the same graph");
    }

    if(x.dtype() != y.dtype())
    {
        throw std::invalid_argument(
            "sum_fiber: input and output tensors must have the same dtype");
    }

    if(axis < 0 || axis >= x.ndim())
    {
        throw std::invalid_argument(
            "sum_fiber: axis out of bounds");
    }

    if(batch_ndim < 0 || axis + batch_ndim > x.ndim())
    {
        throw std::invalid_argument(
            "sum_fiber: invalid batch_ndim");
    }

    OpAttrs attrs = ReductionAttrs{alpha, beta, axis, batch_ndim, redux};
    x.graph().add_op(
        OpType::SUM_FIBER,
        attrs,
        {&x, &y},
        {&y}
    );
}

//! Sum along slices: y = alpha * sum_slice(x) + beta * y
void sum_slice(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& y,
    Index axis,
    int redux,
    Scalar alpha,
    Scalar beta)
{
    if(&x.graph() != &y.graph())
    {
        throw std::invalid_argument(
            "sum_slice: tensors must belong to the same graph");
    }

    if(x.dtype() != y.dtype())
    {
        throw std::invalid_argument(
            "sum_slice: input and output tensors must have the same dtype");
    }

    if(axis < 0 || axis >= x.ndim())
    {
        throw std::invalid_argument(
            "sum_slice: axis out of bounds");
    }

    OpAttrs attrs = ReductionAttrs{alpha, beta, axis, 0, redux};  // batch_ndim = 0 for slice
    x.graph().add_op(
        OpType::SUM_SLICE,
        attrs,
        {&x, &y},
        {&y}
    );
}

//! Euclidean norm: y = alpha * norm(x) + beta * y
void norm(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& y,
    Scalar alpha,
    Scalar beta)
{
    if(&x.graph() != &y.graph())
    {
        throw std::invalid_argument(
            "norm: tensors must belong to the same graph");
    }

    if(y.ndim() != 1 || y.shape()[0] != 1)
    {
        throw std::invalid_argument(
            "norm: output tensor must be scalar (shape [1])");
    }

    if(x.dtype() != y.dtype())
    {
        throw std::invalid_argument(
            "norm: input and output tensors must have the same dtype");
    }

    OpAttrs attrs = TotalSumAttrs{alpha, beta};
    x.graph().add_op(
        OpType::NORM,
        attrs,
        {&x, &y},
        {&y}
    );
}

//! Log sum exp along axis: y = log(sum(exp(x)))
void logsumexp(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& y,
    Index axis)
{
    if(&x.graph() != &y.graph())
    {
        throw std::invalid_argument(
            "logsumexp: tensors must belong to the same graph");
    }

    if(x.dtype() != y.dtype())
    {
        throw std::invalid_argument(
            "logsumexp: input and output tensors must have the same dtype");
    }

    if(axis < 0 || axis >= x.ndim())
    {
        throw std::invalid_argument(
            "logsumexp: axis out of bounds");
    }

    OpAttrs attrs = LogSumExpAttrs{1.0, 0.0, axis};  // alpha=1, beta=0
    x.graph().add_op(
        OpType::LOGSUMEXP,
        attrs,
        {&x},
        {&y}
    );
}

//! Max and sum of exponents along axis: y = max + log(sum(exp(x - max)))
void maxsumexp(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& y,
    Index axis,
    int redux)
{
    if(&x.graph() != &y.graph())
    {
        throw std::invalid_argument(
            "maxsumexp: tensors must belong to the same graph");
    }

    if(x.dtype() != y.dtype())
    {
        throw std::invalid_argument(
            "maxsumexp: input and output tensors must have the same dtype");
    }

    if(axis < 0 || axis >= x.ndim())
    {
        throw std::invalid_argument(
            "maxsumexp: axis out of bounds");
    }

    OpAttrs attrs = LogSumExpAttrs{1.0, 0.0, axis};  // alpha=1, beta=0, axis
    x.graph().add_op(
        OpType::MAXSUMEXP,
        attrs,
        {&x},
        {&y}
    );
}

//! Sum of products along fibers: y = alpha * sum_fiber(x1 * x2) + beta * y
void sumprod_fiber(
    LogicalGraph::TensorNode& x1,
    LogicalGraph::TensorNode& x2,
    LogicalGraph::TensorNode& y,
    Index axis,
    int redux,
    Scalar alpha,
    Scalar beta)
{
    if(&x1.graph() != &x2.graph() || &x1.graph() != &y.graph())
    {
        throw std::invalid_argument(
            "sumprod_fiber: tensors must belong to the same graph");
    }

    if(x1.dtype() != x2.dtype() || x1.dtype() != y.dtype())
    {
        throw std::invalid_argument(
            "sumprod_fiber: all tensors must have the same dtype");
    }

    if(x1.shape() != x2.shape())
    {
        throw std::invalid_argument(
            "sumprod_fiber: x1 and x2 must have the same shape");
    }

    if(axis < 0 || axis >= x1.ndim())
    {
        throw std::invalid_argument(
            "sumprod_fiber: axis out of bounds");
    }

    OpAttrs attrs = ReductionAttrs{alpha, beta, axis, 0, redux};  // batch_ndim = 0 for fiber
    x1.graph().add_op(
        OpType::SUMPROD_FIBER,
        attrs,
        {&x1, &x2, &y},
        {&y}
    );
}

//! Sum of products along slices: y = alpha * sum_slice(x1 * x2) + beta * y
void sumprod_slice(
    LogicalGraph::TensorNode& x1,
    LogicalGraph::TensorNode& x2,
    LogicalGraph::TensorNode& y,
    Index axis,
    int redux,
    Scalar alpha,
    Scalar beta)
{
    if(&x1.graph() != &x2.graph() || &x1.graph() != &y.graph())
    {
        throw std::invalid_argument(
            "sumprod_slice: tensors must belong to the same graph");
    }

    if(x1.dtype() != x2.dtype() || x1.dtype() != y.dtype())
    {
        throw std::invalid_argument(
            "sumprod_slice: all tensors must have the same dtype");
    }

    if(x1.shape() != x2.shape())
    {
        throw std::invalid_argument(
            "sumprod_slice: x1 and x2 must have the same shape");
    }

    if(axis < 0 || axis >= x1.ndim())
    {
        throw std::invalid_argument(
            "sumprod_slice: axis out of bounds");
    }

    OpAttrs attrs = ReductionAttrs{alpha, beta, axis, 0, redux};  // batch_ndim = 0 for slice
    x1.graph().add_op(
        OpType::SUMPROD_SLICE,
        attrs,
        {&x1, &x2, &y},
        {&y}
    );
}

//! Norm along fibers: y = alpha * norm_fiber(x) + beta * y
void norm_fiber(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& y,
    Index axis,
    Index batch_ndim,
    int redux,
    Scalar alpha,
    Scalar beta)
{
    if(&x.graph() != &y.graph())
    {
        throw std::invalid_argument(
            "norm_fiber: tensors must belong to the same graph");
    }

    if(x.dtype() != y.dtype())
    {
        throw std::invalid_argument(
            "norm_fiber: input and output tensors must have the same dtype");
    }

    if(axis < 0 || axis >= x.ndim())
    {
        throw std::invalid_argument(
            "norm_fiber: axis out of bounds");
    }

    if(batch_ndim < 0 || axis + batch_ndim > x.ndim())
    {
        throw std::invalid_argument(
            "norm_fiber: invalid batch_ndim");
    }

    OpAttrs attrs = ReductionAttrs{alpha, beta, axis, batch_ndim, redux};
    x.graph().add_op(
        OpType::NORM_FIBER,
        attrs,
        {&x, &y},
        {&y}
    );
}

//! Norm along fibers (in-place): y = alpha * norm_fiber(x) + beta * y
void norm_fiber_inplace(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& y,
    Index axis,
    Index batch_ndim,
    int redux,
    Scalar alpha,
    Scalar beta)
{
    if(&x.graph() != &y.graph())
    {
        throw std::invalid_argument(
            "norm_fiber_inplace: tensors must belong to the same graph");
    }

    if(x.dtype() != y.dtype())
    {
        throw std::invalid_argument(
            "norm_fiber_inplace: input and output tensors must have the same dtype");
    }

    if(axis < 0 || axis >= x.ndim())
    {
        throw std::invalid_argument(
            "norm_fiber_inplace: axis out of bounds");
    }

    if(batch_ndim < 0 || axis + batch_ndim > x.ndim())
    {
        throw std::invalid_argument(
            "norm_fiber_inplace: invalid batch_ndim");
    }

    OpAttrs attrs = ReductionAttrs{alpha, beta, axis, batch_ndim, redux};
    x.graph().add_op(
        OpType::NORM_FIBER_INPLACE,
        attrs,
        {&x, &y},
        {&y}
    );
}

//! Norm along slices: y = alpha * norm_slice(x) + beta * y
void norm_slice(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& y,
    Index axis,
    int redux,
    Scalar alpha,
    Scalar beta)
{
    if(&x.graph() != &y.graph())
    {
        throw std::invalid_argument(
            "norm_slice: tensors must belong to the same graph");
    }

    if(x.dtype() != y.dtype())
    {
        throw std::invalid_argument(
            "norm_slice: input and output tensors must have the same dtype");
    }

    if(axis < 0 || axis >= x.ndim())
    {
        throw std::invalid_argument(
            "norm_slice: axis out of bounds");
    }

    OpAttrs attrs = ReductionAttrs{alpha, beta, axis, 0, redux};  // batch_ndim = 0 for slice
    x.graph().add_op(
        OpType::NORM_SLICE,
        attrs,
        {&x, &y},
        {&y}
    );
}

//! Norm along slices (in-place): y = alpha * norm_slice(x) + beta * y
void norm_slice_inplace(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& y,
    Index axis,
    int redux,
    Scalar alpha,
    Scalar beta)
{
    if(&x.graph() != &y.graph())
    {
        throw std::invalid_argument(
            "norm_slice_inplace: tensors must belong to the same graph");
    }

    if(x.dtype() != y.dtype())
    {
        throw std::invalid_argument(
            "norm_slice_inplace: input and output tensors must have the same dtype");
    }

    if(axis < 0 || axis >= x.ndim())
    {
        throw std::invalid_argument(
            "norm_slice_inplace: axis out of bounds");
    }

    OpAttrs attrs = ReductionAttrs{alpha, beta, axis, 0, redux};  // batch_ndim = 0 for slice
    x.graph().add_op(
        OpType::NORM_SLICE_INPLACE,
        attrs,
        {&x, &y},
        {&y}
    );
}

//! 2D Convolution forward: Y = alpha * conv2d(X, C) + beta * Y
void conv2d_inplace(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& c,
    LogicalGraph::TensorNode& y,
    Scalar alpha,
    Scalar beta,
    std::array<Index, 2> padding,
    std::array<Index, 2> stride,
    std::array<Index, 2> dilation)
{
    if(&x.graph() != &c.graph() || &x.graph() != &y.graph())
    {
        throw std::invalid_argument(
            "conv2d_inplace: tensors must belong to the same graph");
    }

    if(x.dtype() != c.dtype() || x.dtype() != y.dtype())
    {
        throw std::invalid_argument(
            "conv2d_inplace: all tensors must have the same dtype");
    }

    OpAttrs attrs = Conv2dAttrs{alpha, beta, padding, stride, dilation};
    x.graph().add_op(
        OpType::CONV2D_INPLACE,
        attrs,
        {&x, &c, &y},
        {&y}
    );
}

//! 2D Convolution backward w.r.t. input: dX = alpha * conv2d_bwd_input(dY, C) + beta * dX
void conv2d_bwd_input_inplace(
    LogicalGraph::TensorNode& dy,
    LogicalGraph::TensorNode& c,
    LogicalGraph::TensorNode& dx,
    Scalar alpha,
    Scalar beta,
    std::array<Index, 2> padding,
    std::array<Index, 2> stride,
    std::array<Index, 2> dilation)
{
    if(&dy.graph() != &c.graph() || &dy.graph() != &dx.graph())
    {
        throw std::invalid_argument(
            "conv2d_bwd_input_inplace: tensors must belong to the same graph");
    }

    if(dy.dtype() != c.dtype() || dy.dtype() != dx.dtype())
    {
        throw std::invalid_argument(
            "conv2d_bwd_input_inplace: all tensors must have the same dtype");
    }

    OpAttrs attrs = Conv2dAttrs{alpha, beta, padding, stride, dilation};
    dy.graph().add_op(
        OpType::CONV2D_BWD_INPUT_INPLACE,
        attrs,
        {&dy, &c, &dx},
        {&dx}
    );
}

//! 2D Convolution backward w.r.t. weights: dC = alpha * conv2d_bwd_weight(X, dY) + beta * dC
void conv2d_bwd_weight_inplace(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& dy,
    LogicalGraph::TensorNode& dc,
    Scalar alpha,
    Scalar beta,
    std::array<Index, 2> padding,
    std::array<Index, 2> stride,
    std::array<Index, 2> dilation)
{
    if(&x.graph() != &dy.graph() || &x.graph() != &dc.graph())
    {
        throw std::invalid_argument(
            "conv2d_bwd_weight_inplace: tensors must belong to the same graph");
    }

    if(x.dtype() != dy.dtype() || x.dtype() != dc.dtype())
    {
        throw std::invalid_argument(
            "conv2d_bwd_weight_inplace: all tensors must have the same dtype");
    }

    OpAttrs attrs = Conv2dAttrs{alpha, beta, padding, stride, dilation};
    x.graph().add_op(
        OpType::CONV2D_BWD_WEIGHT_INPLACE,
        attrs,
        {&x, &dy, &dc},
        {&dc}
    );
}

//! Scale operation: y = alpha * x
LogicalGraph::TensorNode& scale(
    LogicalGraph::TensorNode& x,
    const std::string& output_name,
    Scalar alpha)
{
    std::vector<Index> output_shape = x.shape();
    LogicalGraph::TensorNode& output = x.graph().tensor(
        std::move(output_shape),
        output_name,
        x.dtype());

    OpAttrs attrs = ScaleAttrs{alpha};
    x.graph().add_op(
        OpType::SCALE,
        attrs,
        {&x},
        {&output}
    );

    return output;
}

//! Scale in-place: x = alpha * x
void scale_inplace(
    LogicalGraph::TensorNode& x,
    Scalar alpha)
{
    OpAttrs attrs = ScaleAttrs{alpha};
    x.graph().add_op(
        OpType::SCALE_INPLACE,
        attrs,
        {&x},
        {&x}
    );
}

//! Compute embedding output shape
std::vector<Index> compute_embedding_output_shape(
    const LogicalGraph::TensorNode& index,
    const LogicalGraph::TensorNode& vocab,
    Index axis)
{
    if(vocab.ndim() != 2)
    {
        throw std::invalid_argument(
            "embedding: vocab tensor must be 2D");
    }

    std::vector<Index> output_shape = index.shape();
    output_shape.insert(output_shape.begin() + axis, vocab.shape()[0]);

    return output_shape;
}

//! Embedding lookup: y = embedding(x, vocab)
LogicalGraph::TensorNode& embedding(
    LogicalGraph::TensorNode& index,
    LogicalGraph::TensorNode& vocab,
    const std::string& output_name,
    Index axis)
{
    if(&index.graph() != &vocab.graph())
    {
        throw std::invalid_argument(
            "embedding: tensors must belong to the same graph");
    }

    if(index.dtype() != DataType::INT64)
    {
        throw std::invalid_argument(
            "embedding: index tensor must have int64 dtype");
    }

    if(axis < 0 || axis > index.ndim())
    {
        throw std::invalid_argument(
            "embedding: axis out of bounds");
    }

    std::vector<Index> output_shape = compute_embedding_output_shape(
        index, vocab, axis);

    LogicalGraph::TensorNode& output = index.graph().tensor(
        std::move(output_shape),
        output_name,
        vocab.dtype());

    OpAttrs attrs = EmbeddingAttrs{axis};
    index.graph().add_op(
        OpType::EMBEDDING,
        attrs,
        {&index, &vocab},
        {&output}
    );

    return output;
}

//! Embedding backward: vocab += embedding_backward(embed, index, vocab)
void embedding_backward(
    LogicalGraph::TensorNode& embed,
    LogicalGraph::TensorNode& index,
    LogicalGraph::TensorNode& vocab,
    Index axis)
{
    if(&embed.graph() != &index.graph() || &embed.graph() != &vocab.graph())
    {
        throw std::invalid_argument(
            "embedding_backward: tensors must belong to the same graph");
    }

    if(index.dtype() != DataType::INT64)
    {
        throw std::invalid_argument(
            "embedding_backward: index tensor must have int64 dtype");
    }

    if(embed.dtype() != vocab.dtype())
    {
        throw std::invalid_argument(
            "embedding_backward: embed and vocab must have the same dtype");
    }

    if(axis < 0 || axis >= embed.ndim())
    {
        throw std::invalid_argument(
            "embedding_backward: axis out of bounds");
    }

    OpAttrs attrs = EmbeddingAttrs{axis};
    embed.graph().add_op(
        OpType::EMBEDDING_BACKWARD,
        attrs,
        {&embed, &index, &vocab},
        {&vocab}
    );
}

//! Validate inputs for gemm operation
void validate_gemm_inputs(
    LogicalGraph::TensorNode& a,
    LogicalGraph::TensorNode& b,
    LogicalGraph& expected_graph)
{
    // Validate inputs belong to the same graph
    if(&a.graph() != &expected_graph || &b.graph() != &expected_graph)
    {
        throw std::invalid_argument(
            "gemm: input tensors must belong to the same graph");
    }

    // Validate input dtypes match
    if(a.dtype() != b.dtype())
    {
        throw std::invalid_argument(
            "gemm: input tensors must have the same dtype");
    }
}

//! Tensor contraction creating new output: C = alpha * op(A) @ op(B)
LogicalGraph::TensorNode& gemm(
    LogicalGraph::TensorNode& a,
    LogicalGraph::TensorNode& b,
    const std::string& output_name,
    Scalar alpha,
    bool trans_a,
    bool trans_b,
    Index ndim,
    Index batch_ndim)
{
    validate_gemm_inputs(a, b, a.graph());

    // Compute output specification
    std::vector<Index> output_shape = compute_gemm_output_shape(
        a, b, trans_a, trans_b, ndim, batch_ndim
    );

    // Create output tensor
    LogicalGraph::TensorNode& output = a.graph().tensor(
        std::move(output_shape),
        output_name,
        a.dtype());

    // Create operation attributes (beta = 0 for new output)
    OpAttrs attrs = GemmAttrs{trans_a, trans_b, alpha, 0.0, ndim, batch_ndim};

    // Add operation to graph using public builder API
    a.graph().add_op(
        OpType::GEMM,
        attrs,
        {&a, &b},
        {&output}
    );

    return output;
}

//! Tensor contraction with accumulation: C = alpha * op(A) @ op(B) + beta * C
void gemm(
    LogicalGraph::TensorNode& a,
    LogicalGraph::TensorNode& b,
    LogicalGraph::TensorNode& c,
    Scalar alpha,
    Scalar beta,
    bool trans_a,
    bool trans_b,
    Index ndim,
    Index batch_ndim)
{
    validate_gemm_inputs(a, b, a.graph());

    // Validate c belongs to the same graph
    if(&c.graph() != &a.graph())
    {
        throw std::invalid_argument(
            "gemm: tensor c must belong to the same graph as a and b");
    }

    // Validate c dtype matches
    if(c.dtype() != a.dtype())
    {
        throw std::invalid_argument(
            "gemm: tensor c must have the same dtype as a and b");
    }

    // Compute expected output shape and validate against c
    std::vector<Index> expected_shape = compute_gemm_output_shape(
        a, b, trans_a, trans_b, ndim, batch_ndim
    );

    if(c.shape() != expected_shape)
    {
        throw std::invalid_argument(
            "gemm: tensor c has incompatible shape for accumulation");
    }

    // Create operation attributes
    OpAttrs attrs = GemmAttrs{trans_a, trans_b, alpha, beta, ndim, batch_ndim};

    // Add operation to graph
    a.graph().add_op(
        OpType::GEMM,
        attrs,
        {&a, &b, &c},
        {&c}
    );
}

} // namespace nntile::graph
