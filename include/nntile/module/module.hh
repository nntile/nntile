/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/module/module.hh
 * Base Module class for neural network modules.
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <memory>
#include <sstream>
#include <string>
#include <vector>

// Include NNTile headers
#include <nntile/graph.hh>

namespace nntile::module
{

//! Base class for all neural network modules
//!
//! Similar to PyTorch's nn.Module, provides:
//! 1. Forward/backward graph building interface
//! 2. Parameter registration and iteration
//! 3. Submodule composition
//! 4. Named access to parameters and submodules
//!
//! Subclasses should:
//! 1. Call register_parameter() for learnable tensors
//! 2. Call register_buffer() for non-learnable state tensors
//! 3. Call register_module() for child modules
//! 4. Override build_forward() and build_backward()
class Module
{
protected:
    //! Module name (used for generating tensor names)
    std::string name_;

    //! Registered parameters (tensors that need gradients)
    //! Pair of (local_name, tensor_pointer)
    std::vector<std::pair<std::string, graph::TensorNode*>> parameters_;

    //! Registered buffers (tensors that don't need gradients)
    std::vector<std::pair<std::string, graph::TensorNode*>> buffers_;

    //! Child modules
    std::vector<std::pair<std::string, Module*>> submodules_;

    //! Input tensor from last build_forward call
    graph::TensorNode* input_tensor_ = nullptr;

    //! Output tensor from last build_forward call
    graph::TensorNode* output_tensor_ = nullptr;

public:
    //! Constructor
    //! @param name Module name (used for generating unique tensor names)
    explicit Module(const std::string& name);

    //! Virtual destructor for proper cleanup of derived classes
    virtual ~Module() = default;

    // Disable copy (modules often hold pointers to graph elements)
    Module(const Module&) = delete;
    Module& operator=(const Module&) = delete;

    // Allow move
    Module(Module&&) = default;
    Module& operator=(Module&&) = default;

    // -----------------------------------------------------------------
    // Graph Building Interface (to be overridden by subclasses)
    // -----------------------------------------------------------------

    //! Build forward operations into the graph
    //!
    //! This method should:
    //! 1. Create parameter tensors if not already created
    //! 2. Create output tensor
    //! 3. Add forward operations to the graph
    //!
    //! @param graph The logical graph to add operations to
    //! @param input The input tensor
    //! @return Reference to output tensor
    virtual graph::TensorNode& build_forward(
        graph::LogicalGraph& graph,
        graph::TensorNode& input) = 0;

    //! Build backward operations using gradient registry
    //!
    //! This method should:
    //! 1. Get output gradient from registry
    //! 2. Compute parameter gradients (accumulating if shared)
    //! 3. Compute input gradient for upstream modules
    //! 4. Register all computed gradients
    //!
    //! @param graph The logical graph to add operations to
    //! @param grad_reg Gradient registry tracking tensor->gradient mappings
    virtual void build_backward(
        graph::LogicalGraph& graph,
        graph::GradientRegistry& grad_reg) = 0;

    // -----------------------------------------------------------------
    // Parameter/Buffer Registration (called by subclasses)
    // -----------------------------------------------------------------

    //! Register a parameter tensor (will be included in parameter iteration)
    //! @param local_name Local name within this module (e.g., "weight")
    //! @param tensor Pointer to the parameter tensor
    void register_parameter(const std::string& local_name,
                           graph::TensorNode* tensor);

    //! Register a buffer tensor (non-trainable state)
    //! @param local_name Local name within this module
    //! @param tensor Pointer to the buffer tensor
    void register_buffer(const std::string& local_name,
                        graph::TensorNode* tensor);

    //! Register a child module
    //! @param local_name Local name for the submodule
    //! @param module Pointer to the child module (not owned)
    void register_module(const std::string& local_name, Module* module);

    // -----------------------------------------------------------------
    // Parameter Access (for optimizers)
    // -----------------------------------------------------------------

    //! Get all parameters (this module only, not submodules)
    std::vector<graph::TensorNode*> parameters() const;

    //! Get all parameters with local names (this module only)
    const std::vector<std::pair<std::string, graph::TensorNode*>>&
        named_parameters() const;

    //! Get all parameters recursively (including submodules)
    std::vector<graph::TensorNode*> parameters_recursive() const;

    //! Get all parameters with full qualified names recursively
    //! Names are formatted as "module_name.submodule_name.param_name"
    std::vector<std::pair<std::string, graph::TensorNode*>>
        named_parameters_recursive() const;

    // -----------------------------------------------------------------
    // Buffer Access
    // -----------------------------------------------------------------

    //! Get all buffers (this module only)
    std::vector<graph::TensorNode*> buffers() const;

    //! Get all buffers with local names (this module only)
    const std::vector<std::pair<std::string, graph::TensorNode*>>&
        named_buffers() const;

    // -----------------------------------------------------------------
    // Gradient Access (for optimizers after backward)
    // -----------------------------------------------------------------

    //! Get parameter-gradient pairs from registry (this module only)
    //! @param grad_reg The gradient registry after backward pass
    //! @return Vector of (parameter, gradient) pairs
    std::vector<std::pair<graph::TensorNode*, graph::TensorNode*>>
        parameter_gradients(const graph::GradientRegistry& grad_reg) const;

    //! Get parameter-gradient pairs recursively (including submodules)
    std::vector<std::pair<graph::TensorNode*, graph::TensorNode*>>
        parameter_gradients_recursive(
            const graph::GradientRegistry& grad_reg) const;

    // -----------------------------------------------------------------
    // Module Hierarchy
    // -----------------------------------------------------------------

    //! Get child modules (direct children only)
    std::vector<Module*> children() const;

    //! Get named children
    const std::vector<std::pair<std::string, Module*>>& named_children() const;

    //! Get all modules recursively (including self, depth-first)
    std::vector<Module*> modules() const;

    // -----------------------------------------------------------------
    // Tensor Accessors
    // -----------------------------------------------------------------

    //! Get input tensor from last build_forward call
    graph::TensorNode* input_tensor() const { return input_tensor_; }

    //! Get output tensor from last build_forward call
    graph::TensorNode* output_tensor() const { return output_tensor_; }

    // -----------------------------------------------------------------
    // Name Access
    // -----------------------------------------------------------------

    //! Get module name
    const std::string& name() const { return name_; }

    //! Generate full tensor name: "module_name_local_name"
    std::string tensor_name(const std::string& local_name) const;

    //! Generate gradient tensor name: "module_name_local_name_grad"
    std::string grad_name(const std::string& local_name) const;

    // -----------------------------------------------------------------
    // String Representation
    // -----------------------------------------------------------------

    //! Get string representation of module (non-recursive)
    //! Subclasses can override to add module-specific info
    virtual std::string repr() const;

    //! Get full string representation with module hierarchy
    //! Shows all submodules with indentation
    std::string to_string() const;

    //! Print module hierarchy to stdout
    void print() const;

protected:
    //! Helper to collect parameters recursively
    void collect_parameters_recursive(
        const std::string& prefix,
        std::vector<std::pair<std::string, graph::TensorNode*>>& result) const;

    //! Helper to collect modules recursively
    void collect_modules_recursive(std::vector<Module*>& result) const;

    //! Helper for to_string with indentation
    void to_string_recursive(std::ostringstream& ss,
                             const std::string& indent) const;
};

} // namespace nntile::module
