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
Module::Module(graph::NNGraph& graph, const std::string& name)
    : graph_(graph)
    , name_(name)
{
}

// -----------------------------------------------------------------
// Parameter/Buffer Registration
// -----------------------------------------------------------------

void Module::register_parameter(const std::string& local_name,
                                graph::NNGraphTensorNode* tensor)
{
    if(tensor == nullptr)
    {
        throw std::invalid_argument(
            "Module::register_parameter: tensor is nullptr");
    }
    parameters_.emplace_back(local_name, tensor);
}

void Module::register_buffer(const std::string& local_name,
                             graph::NNGraphTensorNode* tensor)
{
    if(tensor == nullptr)
    {
        throw std::invalid_argument(
            "Module::register_buffer: tensor is nullptr");
    }
    graph_.set_requires_grad(*tensor, false);
    buffers_.emplace_back(local_name, tensor);
}

void Module::register_module(const std::string& local_name, Module* module)
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

std::vector<graph::NNGraphTensorNode*> Module::parameters() const
{
    std::vector<graph::NNGraphTensorNode*> result;
    result.reserve(parameters_.size());
    for(const auto& [name, tensor] : parameters_)
    {
        result.push_back(tensor);
    }
    return result;
}

const std::vector<std::pair<std::string, graph::NNGraphTensorNode*>>&
Module::named_parameters() const
{
    return parameters_;
}

std::vector<graph::NNGraphTensorNode*> Module::parameters_recursive() const
{
    std::vector<graph::NNGraphTensorNode*> result;

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

std::vector<std::pair<std::string, graph::NNGraphTensorNode*>>
Module::named_parameters_recursive() const
{
    std::vector<std::pair<std::string, graph::NNGraphTensorNode*>> result;
    collect_parameters_recursive(name_, result);
    return result;
}

void Module::collect_parameters_recursive(
    const std::string& prefix,
    std::vector<std::pair<std::string, graph::NNGraphTensorNode*>>& result) const
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

std::vector<graph::NNGraphTensorNode*> Module::buffers() const
{
    std::vector<graph::NNGraphTensorNode*> result;
    result.reserve(buffers_.size());
    for(const auto& [name, tensor] : buffers_)
    {
        result.push_back(tensor);
    }
    return result;
}

const std::vector<std::pair<std::string, graph::NNGraphTensorNode*>>&
Module::named_buffers() const
{
    return buffers_;
}

// -----------------------------------------------------------------
// Gradient Access
// -----------------------------------------------------------------

std::vector<std::pair<graph::NNGraphTensorNode*,
                      graph::LogicalGraph::TensorNode*>>
Module::parameter_gradients() const
{
    std::vector<std::pair<graph::NNGraphTensorNode*,
                          graph::LogicalGraph::TensorNode*>> result;
    result.reserve(parameters_.size());

    for(const auto& [name, param] : parameters_)
    {
        graph::LogicalGraph::TensorNode* grad = param->grad();
        if(grad != nullptr)
        {
            result.emplace_back(param, grad);
        }
    }

    return result;
}

std::vector<std::pair<graph::NNGraphTensorNode*,
                      graph::LogicalGraph::TensorNode*>>
Module::parameter_gradients_recursive() const
{
    std::vector<std::pair<graph::NNGraphTensorNode*,
                          graph::LogicalGraph::TensorNode*>> result;

    // Add own parameter gradients
    for(const auto& [name, param] : parameters_)
    {
        graph::LogicalGraph::TensorNode* grad = param->grad();
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

std::vector<Module*> Module::children() const
{
    std::vector<Module*> result;
    result.reserve(submodules_.size());
    for(const auto& [name, module] : submodules_)
    {
        result.push_back(module);
    }
    return result;
}

const std::vector<std::pair<std::string, Module*>>&
Module::named_children() const
{
    return submodules_;
}

std::vector<Module*> Module::modules() const
{
    std::vector<Module*> result;
    collect_modules_recursive(result);
    return result;
}

void Module::collect_modules_recursive(std::vector<Module*>& result) const
{
    // Add self (const_cast needed because we return non-const pointers)
    result.push_back(const_cast<Module*>(this));

    // Recurse into submodules
    for(const auto& [name, module] : submodules_)
    {
        module->collect_modules_recursive(result);
    }
}

// -----------------------------------------------------------------
// Name Helpers
// -----------------------------------------------------------------

std::string Module::tensor_name(const std::string& local_name) const
{
    return name_ + "_" + local_name;
}

std::string Module::grad_name(const std::string& local_name) const
{
    return name_ + "_" + local_name + "_grad";
}

// -----------------------------------------------------------------
// String Representation
// -----------------------------------------------------------------

std::string Module::repr() const
{
    return name_ + "()";
}

std::string Module::to_string() const
{
    std::ostringstream ss;
    to_string_recursive(ss, "");
    return ss.str();
}

void Module::print() const
{
    std::cout << to_string() << std::endl;
}

void Module::to_string_recursive(std::ostringstream& ss,
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
