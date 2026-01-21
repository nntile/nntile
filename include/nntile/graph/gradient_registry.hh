/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/gradient_registry.hh
 * GradientRegistry class for tracking tensor gradients during backward pass.
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <map>
#include <string>

// Include NNTile headers
#include <nntile/graph/tensor_node.hh>

namespace nntile::graph
{

// Forward declarations
class LogicalGraph;

//! Registry that maps tensors to their gradient tensors
//!
//! This class is used during backward pass construction to:
//! 1. Track which tensors have gradient tensors
//! 2. Enable gradient accumulation for shared weights
//! 3. Allow composable module backward passes
//!
//! Usage pattern:
//! @code
//! GradientRegistry grad_reg;
//! // Set the initial gradient (e.g., from loss)
//! grad_reg.set_grad("output", &grad_output_tensor);
//! // Build backward passes - each module updates the registry
//! module2.build_backward(graph, grad_reg);
//! module1.build_backward(graph, grad_reg);
//! @endcode
class GradientRegistry
{
private:
    // Map from tensor name to its gradient tensor
    std::map<std::string, TensorNode*> grad_map_;

public:
    GradientRegistry() = default;

    // -----------------------------------------------------------------
    // Query methods
    // -----------------------------------------------------------------

    //! Check if a gradient is registered for the given tensor name
    bool has_grad(const std::string& tensor_name) const;

    //! Check if a gradient is registered for the given tensor
    bool has_grad(const TensorNode& tensor) const;

    //! Get gradient tensor for the given tensor name
    //! @return Pointer to gradient tensor, or nullptr if not registered
    TensorNode* get_grad(const std::string& tensor_name) const;

    //! Get gradient tensor for the given tensor
    //! @return Pointer to gradient tensor, or nullptr if not registered
    TensorNode* get_grad(const TensorNode& tensor) const;

    // -----------------------------------------------------------------
    // Registration methods
    // -----------------------------------------------------------------

    //! Register a gradient tensor for the given tensor name
    //! @param tensor_name Name of the tensor to register gradient for
    //! @param grad_tensor Pointer to the gradient tensor
    //! @throws std::runtime_error if gradient already registered (use
    //!         has_grad to check first, or use get_or_create_grad for
    //!         accumulation)
    void set_grad(const std::string& tensor_name, TensorNode* grad_tensor);

    //! Register a gradient tensor for the given tensor
    void set_grad(const TensorNode& tensor, TensorNode* grad_tensor);

    // -----------------------------------------------------------------
    // Gradient creation helpers
    // -----------------------------------------------------------------

    //! Get existing gradient tensor or create a new one
    //!
    //! This is the primary method for backward pass construction:
    //! - If gradient already exists, returns it (for accumulation)
    //! - If not, creates a new tensor in the graph and registers it
    //!
    //! @param graph The logical graph to create tensor in (if needed)
    //! @param tensor The tensor to get/create gradient for
    //! @param grad_name Name for the gradient tensor (used only if creating)
    //! @return Reference to the gradient tensor (existing or newly created)
    TensorNode& get_or_create_grad(
        LogicalGraph& graph,
        const TensorNode& tensor,
        const std::string& grad_name);

    //! Check if this is the first gradient contribution for a tensor
    //!
    //! Useful for deciding whether to initialize (beta=0) or accumulate
    //! (beta=1) in GEMM operations.
    //!
    //! @param tensor_name Name of the tensor
    //! @return true if no gradient registered yet
    bool is_first_grad(const std::string& tensor_name) const;

    //! Check if this is the first gradient contribution for a tensor
    bool is_first_grad(const TensorNode& tensor) const;

    // -----------------------------------------------------------------
    // Iteration
    // -----------------------------------------------------------------

    //! Get all registered tensor-gradient pairs
    const std::map<std::string, TensorNode*>& all_grads() const
    {
        return grad_map_;
    }

    //! Number of registered gradients
    size_t size() const { return grad_map_.size(); }

    //! Check if registry is empty
    bool empty() const { return grad_map_.empty(); }
};

} // namespace nntile::graph
