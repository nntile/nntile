/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/logical_graph.cc
 * Implementation of LogicalGraph class.
 *
 * @version 1.1.0
 * */

#include <nntile/graph/logical_graph.hh>
#include <stdexcept>
#include <sstream>

namespace nntile::graph {

//! Logical graph - defines computation without physical details
LogicalGraph::LogicalGraph(const std::string& name)
    : name_(name)
{
}

//! Create a tensor
TensorNode& LogicalGraph::tensor(const TensorSpec& spec, const std::string& name)
{
    // Check name doesn't already exist
    if(tensor_by_name_.count(name) > 0)
    {
        throw std::invalid_argument("LogicalGraph::tensor: tensor '" + name +
                "' already exists");
    }

    // Create TensorNode with unique ID
    auto node = std::make_unique<TensorNode>(next_tensor_id_++, name, spec, this);
    TensorNode* node_ptr = node.get();

    // Store in containers
    tensors_.push_back(std::move(node));
    tensor_by_name_[name] = node_ptr;

    return *node_ptr;
}


//! General matrix multiplication: C = alpha * A @ B + beta * C
TensorNode& LogicalGraph::gemm(
    TensorNode& a,
    TensorNode& b,
    const std::string& output_name,
    Scalar alpha,
    Scalar beta,
    bool trans_a,
    bool trans_b,
    Index ndim,
    Index batch_ndim
)
{
    // Validate input shapes are compatible
    TensorSpec output_spec = compute_gemm_output_spec(a.spec(), b.spec(),
            trans_a, trans_b, ndim, batch_ndim);

    // Create OpNode with GemmAttrs
    OpAttrs attrs = GemmAttrs{trans_a, trans_b, alpha, beta, ndim, batch_ndim};
    auto op = std::make_unique<OpNode>(next_op_id_++, OpType::GEMM, attrs, this);
    OpNode* op_ptr = op.get();

    // Create output TensorNode
    TensorNode& output = create_op_output(*op_ptr, output_spec, output_name);

    // Wire up edges (inputs/outputs/producer/consumers)
    op_ptr->add_input(&a);
    op_ptr->add_input(&b);

    // Store operation
    ops_.push_back(std::move(op));

    return output;
}

//! GeLU activation: y = gelu(x)
TensorNode& LogicalGraph::gelu(
    TensorNode& x,
    const std::string& output_name
) {
    // Output shape = input shape
    TensorSpec output_spec = TensorSpec(x.shape(), x.dtype());

    // Create OpNode with GeluAttrs
    OpAttrs attrs = GeluAttrs{};
    auto op = std::make_unique<OpNode>(next_op_id_++, OpType::GELU, attrs, this);
    OpNode* op_ptr = op.get();

    // Create output TensorNode
    TensorNode& output = create_op_output(*op_ptr, output_spec, output_name);

    // Wire up edges
    op_ptr->add_input(&x);

    // Store operation
    ops_.push_back(std::move(op));

    return output;
}

//! Get tensor by name (returns nullptr if not found)
TensorNode* LogicalGraph::get_tensor(const std::string& name)
{
    auto it = tensor_by_name_.find(name);
    return it != tensor_by_name_.end() ? it->second : nullptr;
}

//! Get tensor by name (returns nullptr if not found)
const TensorNode* LogicalGraph::get_tensor(const std::string& name) const
{
    auto it = tensor_by_name_.find(name);
    return it != tensor_by_name_.end() ? it->second : nullptr;
}

//! Get all tensor names
std::vector<std::string> LogicalGraph::tensor_names() const
{
    std::vector<std::string> names;
    names.reserve(tensor_by_name_.size());
    for(const auto& pair : tensor_by_name_)
    {
        names.push_back(pair.first);
    }
    return names;
}


//! String representation
std::string LogicalGraph::to_string() const
{
    std::stringstream ss;
    ss << "LogicalGraph(name='" << name_ << "', tensors=" << num_tensors()
       << ", ops=" << num_ops() << ")\n";

    ss << "Tensors:\n";
    for(const auto& t : tensors_)
    {
        ss << "  " << t->to_string() << "\n";
    }

    ss << "Operations:\n";
    for(const auto& op : ops_)
    {
        ss << "  " << op->to_string() << "\n";
    }

    return ss.str();
}

//! Internal: create output tensor for an operation
TensorNode& LogicalGraph::create_op_output(
    OpNode& op,
    const TensorSpec& spec,
    const std::string& name
)
{
    // Check name doesn't already exist
    if(tensor_by_name_.count(name) > 0)
    {
        throw std::invalid_argument("LogicalGraph::create_op_output: tensor '" +
                name + "' already exists");
    }

    // Create TensorNode with unique ID
    auto node = std::make_unique<TensorNode>(next_tensor_id_++, name, spec, this);
    TensorNode* node_ptr = node.get();

    // Wire up with operation
    op.add_output(node_ptr);

    // Store in containers
    tensors_.push_back(std::move(node));
    tensor_by_name_[name] = node_ptr;

    return *node_ptr;
}

//! Compute output shape for gemm
TensorSpec LogicalGraph::compute_gemm_output_spec(
    const TensorSpec& a,
    const TensorSpec& b,
    bool trans_a,
    bool trans_b,
    Index ndim,
    Index batch_ndim
)
{
    // For now, support only 2D case (ndim=1, batch_ndim=0)
    // This matches the current implementation
    if(ndim != 1 || batch_ndim != 0)
    {
        throw std::invalid_argument("gemm: only 2D matrices (ndim=1, batch_ndim=0) "
                "are currently supported");
    }
    if(a.ndim() != 2 || b.ndim() != 2)
    {
        throw std::invalid_argument("gemm: tensors must be 2D");
    }

    // A: [M, K] (or [K, M] if trans_a)
    // B: [K, N] (or [N, K] if trans_b)
    // C: [M, N]
    Index M = trans_a ? a.dim(1) : a.dim(0);
    Index K_a = trans_a ? a.dim(0) : a.dim(1);
    Index K_b = trans_b ? b.dim(1) : b.dim(0);
    Index N = trans_b ? b.dim(0) : b.dim(1);

    if(K_a != K_b)
    {
        throw std::invalid_argument("gemm: incompatible shapes - K dimensions "
                "don't match");
    }

    return TensorSpec({M, N}, a.dtype());
}

} // namespace nntile::graph
