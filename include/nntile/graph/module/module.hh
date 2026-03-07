/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/module/module.hh
 * Base Module class for neural network modules.
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

// Include NNTile headers
#include <nntile/graph.hh>
#include <nntile/graph/io/safetensors.hh>

namespace nntile::graph::module
{

//! Base class for all neural network modules (registration, parameters, etc.)
class Module
{
protected:
    //! Pointer to the graph this module belongs to
    NNGraph* graph_;

    //! Module name (used for generating tensor names)
    std::string name_;

    //! Registered parameters (tensors that need gradients)
    //! Pair of (local_name, tensor_pointer)
    std::vector<std::pair<std::string, NNGraph::TensorNode*>> parameters_;

    //! Registered buffers (tensors that don't need gradients)
    std::vector<std::pair<std::string, NNGraph::TensorNode*>> buffers_;

    //! Child modules
    std::vector<std::pair<std::string, Module*>> submodules_;

public:
    //! Constructor
    //! @param graph Pointer to the neural network graph this module belongs to
    //! @param name Module name (used for generating unique tensor names)
    Module(NNGraph* graph, const std::string& name);

    //! Virtual destructor for proper cleanup of derived classes
    virtual ~Module() = default;

    // Disable copy (modules hold references to graph elements)
    Module(const Module&) = delete;
    Module& operator=(const Module&) = delete;

    // Disable move (due to graph reference)
    Module(Module&&) = delete;
    Module& operator=(Module&&) = delete;

    // -----------------------------------------------------------------
    // Graph Access
    // -----------------------------------------------------------------

    //! Get the graph this module belongs to
    NNGraph* graph() { return graph_; }
    const NNGraph* graph() const { return graph_; }

    // -----------------------------------------------------------------
    // Parameter/Buffer Registration (called by subclasses)
    // -----------------------------------------------------------------

    //! Register a parameter tensor (will be included in parameter iteration)
    //! @param local_name Local name within this module (e.g., "weight")
    //! @param tensor Pointer to the parameter tensor
    void register_parameter(const std::string& local_name,
                           NNGraph::TensorNode* tensor);

    //! Register a buffer tensor (non-trainable state)
    //! @param local_name Local name within this module
    //! @param tensor Pointer to the buffer tensor
    void register_buffer(const std::string& local_name,
                        NNGraph::TensorNode* tensor);

    //! Register a child module
    //! @param local_name Local name for the submodule
    //! @param module Pointer to the child module (not owned)
    void register_module(const std::string& local_name, Module* module);

    // -----------------------------------------------------------------
    // Parameter Access (for optimizers)
    // -----------------------------------------------------------------

    //! Get all parameters (this module only, not submodules)
    std::vector<NNGraph::TensorNode*> parameters() const;

    //! Get all parameters with local names (this module only)
    const std::vector<std::pair<std::string, NNGraph::TensorNode*>>&
        named_parameters() const;

    //! Get all parameters recursively (including submodules)
    std::vector<NNGraph::TensorNode*> parameters_recursive() const;

    //! Get all parameters with full qualified names recursively
    //! Names are formatted as "module_name.submodule_name.param_name"
    std::vector<std::pair<std::string, NNGraph::TensorNode*>>
        named_parameters_recursive() const;

    // -----------------------------------------------------------------
    // Buffer Access
    // -----------------------------------------------------------------

    //! Get all buffers (this module only)
    std::vector<NNGraph::TensorNode*> buffers() const;

    //! Get all buffers with local names (this module only)
    const std::vector<std::pair<std::string, NNGraph::TensorNode*>>&
        named_buffers() const;

    // -----------------------------------------------------------------
    // Gradient Access (for optimizers after backward)
    // -----------------------------------------------------------------

    //! Get parameter-gradient pairs from stored grad tensors (this module only)
    //! @return Vector of (parameter, gradient) pairs
    std::vector<std::pair<NNGraph::TensorNode*,
                          NNGraph::TensorNode*>>
        parameter_gradients() const;

    //! Get parameter-gradient pairs recursively (including submodules)
    std::vector<std::pair<NNGraph::TensorNode*,
                          NNGraph::TensorNode*>>
        parameter_gradients_recursive() const;

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
    // Name Access
    // -----------------------------------------------------------------

    //! Get module name
    const std::string& name() const { return name_; }

    //! Generate full tensor name: "module_name_local_name"
    std::string tensor_name(const std::string& local_name) const;

    //! Generate gradient tensor name for a registered parameter:
    //! "module_name_local_name_grad". Use only for module parameters (e.g.
    //! "weight", "bias"), not for external input tensors. For input gradients,
    //! use input_tensor->name() + "_grad".
    std::string grad_name(const std::string& local_name) const;

    // -----------------------------------------------------------------
    // Serialization (NNTile-native SafeTensors)
    // -----------------------------------------------------------------

    //! Save all parameters to a SafeTensors file in NNTile-native layout.
    //! Iterates named_parameters_recursive() and writes each parameter's
    //! bind_hint data. Parameters without a bind_hint are skipped.
    void save(const std::string& path) const;

    //! Load parameters from a SafeTensors file in NNTile-native layout.
    //! Matches tensor names from the file to named_parameters_recursive()
    //! and calls set_bind_hint() on each matched parameter.
    //! @param strict If true (default), throws if any module parameter is
    //!        missing from the file. If false, missing tensors are skipped.
    void load(const std::string& path, bool strict = true);

    // -----------------------------------------------------------------
    // HuggingFace Import/Export
    // -----------------------------------------------------------------

    //! Import weights from an HF-format SafeTensors file.
    //! Default implementation delegates to submodules, appending child
    //! names to the prefix. Leaf modules override to do the actual
    //! data reading and conversion (transpose, reshape, etc.).
    //! @param reader  An already-opened SafeTensorsReader for the HF file.
    //! @param hf_prefix  The HF key prefix for this module's tensors
    //!                   (e.g. "model.layers.0.self_attn").
    virtual void import_hf(const io::SafeTensorsReader& reader,
                           const std::string& hf_prefix);

    //! Export weights to an HF-format SafeTensors file.
    //! Default implementation delegates to submodules. Leaf modules
    //! override to do the reverse conversion and write via the writer.
    virtual void export_hf(io::SafeTensorsWriter& writer,
                           const std::string& hf_prefix) const;

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
        std::vector<std::pair<std::string, NNGraph::TensorNode*>>& result)
            const;

    //! Helper to collect modules recursively
    void collect_modules_recursive(std::vector<Module*>& result) const;

    //! Helper for to_string with indentation
    void to_string_recursive(std::ostringstream& ss,
                             const std::string& indent) const;
};

} // namespace nntile::graph::module
