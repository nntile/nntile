/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/gradient_registry.cc
 * Implementation of GradientRegistry class.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/gradient_registry.hh"

// Include standard headers
#include <stdexcept>

// Include NNTile headers
#include "nntile/graph/logical_graph.hh"

namespace nntile::graph
{

// -----------------------------------------------------------------
// Gradient requirement tracking
// -----------------------------------------------------------------

void GradientRegistry::set_requires_grad(const std::string& tensor_name,
                                          bool requires)
{
    if(requires)
    {
        requires_grad_.insert(tensor_name);
    }
    else
    {
        requires_grad_.erase(tensor_name);
    }
}

void GradientRegistry::set_requires_grad(const TensorNode& tensor, bool requires)
{
    set_requires_grad(tensor.name(), requires);
}

bool GradientRegistry::requires_grad(const std::string& tensor_name) const
{
    // A tensor requires gradient if explicitly marked OR if it already has one
    return requires_grad_.find(tensor_name) != requires_grad_.end() ||
           has_grad(tensor_name);
}

bool GradientRegistry::requires_grad(const TensorNode& tensor) const
{
    return requires_grad(tensor.name());
}

// -----------------------------------------------------------------
// Query methods
// -----------------------------------------------------------------

//! Check if a gradient is registered for the given tensor name
bool GradientRegistry::has_grad(const std::string& tensor_name) const
{
    return grad_map_.find(tensor_name) != grad_map_.end();
}

//! Check if a gradient is registered for the given tensor
bool GradientRegistry::has_grad(const TensorNode& tensor) const
{
    return has_grad(tensor.name());
}

//! Get gradient tensor for the given tensor name
TensorNode* GradientRegistry::get_grad(const std::string& tensor_name) const
{
    auto it = grad_map_.find(tensor_name);
    return it != grad_map_.end() ? it->second : nullptr;
}

//! Get gradient tensor for the given tensor
TensorNode* GradientRegistry::get_grad(const TensorNode& tensor) const
{
    return get_grad(tensor.name());
}

//! Register a gradient tensor for the given tensor name
void GradientRegistry::set_grad(
    const std::string& tensor_name,
    TensorNode* grad_tensor)
{
    if(has_grad(tensor_name))
    {
        throw std::runtime_error(
            "GradientRegistry::set_grad: gradient for tensor '" +
            tensor_name + "' already registered. Use get_or_create_grad "
            "for accumulation.");
    }
    grad_map_[tensor_name] = grad_tensor;
}

//! Register a gradient tensor for the given tensor
void GradientRegistry::set_grad(
    const TensorNode& tensor,
    TensorNode* grad_tensor)
{
    set_grad(tensor.name(), grad_tensor);
}

//! Get existing gradient tensor or create a new one
TensorNode& GradientRegistry::get_or_create_grad(
    LogicalGraph& graph,
    const TensorNode& tensor,
    const std::string& grad_name)
{
    // Check if gradient already exists
    TensorNode* existing = get_grad(tensor);
    if(existing != nullptr)
    {
        return *existing;
    }

    // Create new gradient tensor with same spec as original tensor
    TensorNode& grad_tensor = graph.tensor(tensor.spec(), grad_name);

    // Register it
    grad_map_[tensor.name()] = &grad_tensor;

    return grad_tensor;
}

//! Check if this is the first gradient contribution for a tensor
bool GradientRegistry::is_first_grad(const std::string& tensor_name) const
{
    return !has_grad(tensor_name);
}

//! Check if this is the first gradient contribution for a tensor
bool GradientRegistry::is_first_grad(const TensorNode& tensor) const
{
    return is_first_grad(tensor.name());
}

} // namespace nntile::graph
