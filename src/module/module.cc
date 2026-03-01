/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/module/module.cc
 * Implementation of base Module class.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/module/module.hh"

// Include standard headers
#include <iostream>
#include <stdexcept>

namespace nntile::module
{

//! Constructor
ModuleBase::ModuleBase(graph::NNGraph& graph, const std::string& name)
    : graph_(graph)
    , name_(name)
{
}

// -----------------------------------------------------------------
// Parameter/Buffer Registration
// -----------------------------------------------------------------

void ModuleBase::register_parameter(const std::string& local_name,
                                graph::NNGraph::TensorNode* tensor)
{
    if(tensor == nullptr)
    {
        throw std::invalid_argument(
            "Module::register_parameter: tensor is nullptr");
    }
    parameters_.emplace_back(local_name, tensor);
}

void ModuleBase::register_buffer(const std::string& local_name,
                             graph::NNGraph::TensorNode* tensor)
{
    if(tensor == nullptr)
    {
        throw std::invalid_argument(
            "Module::register_buffer: tensor is nullptr");
    }
    graph_.set_requires_grad(tensor, false);
    buffers_.emplace_back(local_name, tensor);
}

void ModuleBase::register_module(const std::string& local_name,
                                  ModuleBase* module)
{
    if(module == nullptr)
    {
        throw std::invalid_argument(
            "Module::register_module: module is nullptr");
    }
    submodules_.emplace_back(local_name, module);
}

// -----------------------------------------------------------------
// Parameter Access
// -----------------------------------------------------------------

std::vector<graph::NNGraph::TensorNode*> ModuleBase::parameters() const
{
    std::vector<graph::NNGraph::TensorNode*> result;
    result.reserve(parameters_.size());
    for(const auto& [name, tensor] : parameters_)
    {
        result.push_back(tensor);
    }
    return result;
}

const std::vector<std::pair<std::string, graph::NNGraph::TensorNode*>>&
ModuleBase::named_parameters() const
{
    return parameters_;
}

std::vector<graph::NNGraph::TensorNode*> ModuleBase::parameters_recursive() const
{
    std::vector<graph::NNGraph::TensorNode*> result;

    // Add own parameters
    for(const auto& [name, tensor] : parameters_)
    {
        result.push_back(tensor);
    }

    // Add submodule parameters recursively
    for(const auto& [name, module] : submodules_)
    {
        auto sub_params = module->parameters_recursive();
        result.insert(result.end(), sub_params.begin(), sub_params.end());
    }

    return result;
}

std::vector<std::pair<std::string, graph::NNGraph::TensorNode*>>
ModuleBase::named_parameters_recursive() const
{
    std::vector<std::pair<std::string, graph::NNGraph::TensorNode*>> result;
    collect_parameters_recursive(name_, result);
    return result;
}

void ModuleBase::collect_parameters_recursive(
    const std::string& prefix,
    std::vector<std::pair<std::string, graph::NNGraph::TensorNode*>>& result) const
{
    // Add own parameters with prefix
    for(const auto& [local_name, tensor] : parameters_)
    {
        result.emplace_back(prefix + "." + local_name, tensor);
    }

    // Recurse into submodules
    for(const auto& [sub_name, module] : submodules_)
    {
        module->collect_parameters_recursive(prefix + "." + sub_name, result);
    }
}

// -----------------------------------------------------------------
// Buffer Access
// -----------------------------------------------------------------

std::vector<graph::NNGraph::TensorNode*> ModuleBase::buffers() const
{
    std::vector<graph::NNGraph::TensorNode*> result;
    result.reserve(buffers_.size());
    for(const auto& [name, tensor] : buffers_)
    {
        result.push_back(tensor);
    }
    return result;
}

const std::vector<std::pair<std::string, graph::NNGraph::TensorNode*>>&
ModuleBase::named_buffers() const
{
    return buffers_;
}

// -----------------------------------------------------------------
// Gradient Access
// -----------------------------------------------------------------

std::vector<std::pair<graph::NNGraph::TensorNode*,
                      graph::NNGraph::TensorNode*>>
ModuleBase::parameter_gradients() const
{
    std::vector<std::pair<graph::NNGraph::TensorNode*,
                          graph::NNGraph::TensorNode*>> result;
    result.reserve(parameters_.size());

    for(const auto& [name, param] : parameters_)
    {
        graph::NNGraph::TensorNode* grad = param->grad();
        if(grad != nullptr)
        {
            result.emplace_back(param, grad);
        }
    }

    return result;
}

std::vector<std::pair<graph::NNGraph::TensorNode*,
                      graph::NNGraph::TensorNode*>>
ModuleBase::parameter_gradients_recursive() const
{
    std::vector<std::pair<graph::NNGraph::TensorNode*,
                          graph::NNGraph::TensorNode*>> result;

    // Add own parameter gradients
    for(const auto& [name, param] : parameters_)
    {
        graph::NNGraph::TensorNode* grad = param->grad();
        if(grad != nullptr)
        {
            result.emplace_back(param, grad);
        }
    }

    // Add submodule parameter gradients recursively
    for(const auto& [name, module] : submodules_)
    {
        auto sub_grads = module->parameter_gradients_recursive();
        result.insert(result.end(), sub_grads.begin(), sub_grads.end());
    }

    return result;
}

// -----------------------------------------------------------------
// Module Hierarchy
// -----------------------------------------------------------------

std::vector<ModuleBase*> ModuleBase::children() const
{
    std::vector<ModuleBase*> result;
    result.reserve(submodules_.size());
    for(const auto& [name, module] : submodules_)
    {
        result.push_back(module);
    }
    return result;
}

const std::vector<std::pair<std::string, ModuleBase*>>&
ModuleBase::named_children() const
{
    return submodules_;
}

std::vector<ModuleBase*> ModuleBase::modules() const
{
    std::vector<ModuleBase*> result;
    collect_modules_recursive(result);
    return result;
}

void ModuleBase::collect_modules_recursive(std::vector<ModuleBase*>& result) const
{
    // Add self (const_cast needed because we return non-const pointers)
    result.push_back(const_cast<ModuleBase*>(this));

    // Recurse into submodules
    for(const auto& [name, module] : submodules_)
    {
        module->collect_modules_recursive(result);
    }
}

// -----------------------------------------------------------------
// Name Helpers
// -----------------------------------------------------------------

std::string ModuleBase::tensor_name(const std::string& local_name) const
{
    return name_ + "_" + local_name;
}

std::string ModuleBase::grad_name(const std::string& local_name) const
{
    return name_ + "_" + local_name + "_grad";
}

// -----------------------------------------------------------------
// String Representation
// -----------------------------------------------------------------

std::string ModuleBase::repr() const
{
    return name_ + "()";
}

std::string ModuleBase::to_string() const
{
    std::ostringstream ss;
    to_string_recursive(ss, "");
    return ss.str();
}

void ModuleBase::print() const
{
    std::cout << to_string() << std::endl;
}

void ModuleBase::to_string_recursive(std::ostringstream& ss,
                                  const std::string& indent) const
{
    // Print this module
    ss << indent << repr();

    // If we have submodules, print them
    if(!submodules_.empty())
    {
        ss << " {\n";

        // Print parameters first
        for(const auto& [param_name, tensor] : parameters_)
        {
            ss << indent << "  (" << param_name << "): Parameter(";
            if(tensor != nullptr)
            {
                ss << "shape=[";
                const auto& shape = tensor->shape();
                for(size_t i = 0; i < shape.size(); ++i)
                {
                    if(i > 0) ss << ", ";
                    ss << shape[i];
                }
                ss << "]";
            }
            else
            {
                ss << "not created";
            }
            ss << ")\n";
        }

        // Print submodules
        for(const auto& [sub_name, module] : submodules_)
        {
            ss << indent << "  (" << sub_name << "): ";
            module->to_string_recursive(ss, indent + "  ");
        }

        ss << indent << "}";
    }
    else if(!parameters_.empty())
    {
        // No submodules but has parameters
        ss << " {\n";
        for(const auto& [param_name, tensor] : parameters_)
        {
            ss << indent << "  (" << param_name << "): Parameter(";
            if(tensor != nullptr)
            {
                ss << "shape=[";
                const auto& shape = tensor->shape();
                for(size_t i = 0; i < shape.size(); ++i)
                {
                    if(i > 0) ss << ", ";
                    ss << shape[i];
                }
                ss << "]";
            }
            else
            {
                ss << "not created";
            }
            ss << ")\n";
        }
        ss << indent << "}";
    }

    ss << "\n";
}

} // namespace nntile::module
