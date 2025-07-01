/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/autograd/add.cc
 * Add function for autograd
 *
 * @version 1.1.0
 * */

#include "nntile/autograd/add.hh"
#include <stdexcept>

namespace nntile::autograd
{

std::vector<TensorHandle> Add::forward(
    const std::vector<TensorHandle>& inputs)
{
    // Check inputs
    if (inputs.size() != 2) {
        throw std::runtime_error("Add::forward: expected 2 inputs");
    }

    // Save inputs for backward pass
    save_for_backward(inputs);

    // Create output tensor
    // auto output = TensorHandle(inputs[0].shape(), inputs[0].dtype());
    auto output = TensorHandle();

    // Perform forward pass
    // tensor::add(alpha_, inputs[0], beta_, inputs[1], output);

    return {output};
}

std::vector<TensorHandle> Add::backward(
    const std::vector<TensorHandle>& grad_outputs)
{
    // Check gradient outputs
    if (grad_outputs.size() != 1) {
        throw std::runtime_error("Add::backward: expected 1 gradient output");
    }

    // Get saved inputs
    if (saved_tensors_.size() != 2) {
        throw std::runtime_error("Add::backward: expected 2 saved tensors");
    }
    auto& input0 = saved_tensors_[0];
    auto& input1 = saved_tensors_[1];
    auto& input2 = saved_tensors_[2];

    // Create gradient tensors for inputs
    // auto grad_input0 = TensorHandle(input0.shape(), input0.dtype());
    // auto grad_input1 = TensorHandle(input1.shape(), input1.dtype());
    auto grad_input0 = TensorHandle();
    auto grad_input1 = TensorHandle();

    // Compute gradients for inputs
    if (input0.requires_grad()) {
        // tensor::scal(alpha_, grad_outputs[0], grad_input0);
    }
    if (input1.requires_grad()) {
        // tensor::scal(beta_, grad_outputs[0], grad_input1);
    }

    return {grad_input0, grad_input1};
}

AutogradTensor add(
    Scalar alpha,
    const AutogradTensor& a,
    Scalar beta,
    const AutogradTensor& b
)
{
    // Create function
    auto func = std::make_shared<Add>(alpha, beta);

    // Create autograd tensor
    std::vector<AutogradTensor> inputs = {a, b};
    return AutogradTensor(func, inputs);
}

} // namespace nntile::autograd
