/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/autograd/tensor.cc
 * Tensor with autograd support
 *
 * @version 1.1.0
 * */

#include "nntile/autograd/tensor.hh"

namespace nntile::autograd
{

// Constructor is implemented in the header file

// Getter methods are implemented in the header file

void AutogradTensor::backward()
{
    if (!requires_grad_) return;

    // Get all functions in reverse topological order
    std::vector<std::shared_ptr<Function>> functions;
    std::unordered_set<Function*> visited;

    std::function<void(Function*)> visit = [&](Function* fn) {
        if (visited.count(fn)) return;
        visited.insert(fn);
        functions.push_back(std::shared_ptr<Function>(fn));
    };

    if (grad_fn_) {
        visit(grad_fn_.get());
    }

    // Execute backward pass
    for (auto it = functions.rbegin(); it != functions.rend(); ++it) {
        auto& fn = *it;
        auto grad_outputs = fn->backward({grad_});
        // For now, just store the gradients
        // In real implementation, this would accumulate gradients
    }
}

} // namespace nntile::autograd
