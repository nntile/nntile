/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/autograd/tensor.hh
 * Tensor with autograd support
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/autograd/function.hh>
#include <memory>
#include <unordered_set>
#include <functional>
#include <stdexcept>
#include <vector>

namespace nntile::autograd
{

//! Tensor with autograd support
class AutogradTensor
{
private:
    //! Underlying tensor (mock for now)
    TensorHandle tensor_;
    //! Gradient tensor (mock for now)
    TensorHandle grad_;
    //! Function that created this tensor
    std::shared_ptr<Function> grad_fn_;
    //! Whether the tensor requires gradient computation
    bool requires_grad_;

public:
    //! Default constructor
    AutogradTensor(bool requires_grad = true):
        requires_grad_(requires_grad)
    {
        // Create mock tensors
        tensor_ = TensorHandle(requires_grad);
        grad_ = TensorHandle(requires_grad);
    }

    //! Constructor from function and input tensors
    AutogradTensor(const std::shared_ptr<Function>& fn,
                  const std::vector<AutogradTensor>& inputs):
        requires_grad_(fn->requires_grad())
    {
        // Create mock tensors
        tensor_ = TensorHandle(requires_grad_);
        grad_ = TensorHandle(requires_grad_);
        // Set gradient function
        grad_fn_ = fn;
        // Extract underlying tensors from inputs
        std::vector<TensorHandle> input_tensors;
        input_tensors.reserve(inputs.size());
        for (const auto& input : inputs) {
            input_tensors.push_back(input.tensor());
        }
        // Perform forward pass
        auto outputs = fn->forward(input_tensors);
        if (outputs.size() != 1) {
            throw std::runtime_error("AutogradTensor: function must return exactly one tensor");
        }
        tensor_ = outputs[0];
    }

    //! Get underlying tensor
    const TensorHandle& tensor() const { return tensor_; }

    //! Get gradient tensor
    const TensorHandle& grad() const { return grad_; }

    //! Get gradient function
    const std::shared_ptr<Function>& grad_fn() const { return grad_fn_; }

    //! Check if tensor requires gradient
    bool requires_grad() const { return requires_grad_; }

    //! Set gradient function
    void set_grad_fn(const std::shared_ptr<Function>& fn) { grad_fn_ = fn; }

    //! Backward pass
    void backward();

    //! Forward pass operations
    AutogradTensor add(const AutogradTensor& other, double alpha = 1.0, double beta = 1.0);
};

} // namespace nntile::autograd
