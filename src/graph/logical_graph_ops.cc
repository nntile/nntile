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
