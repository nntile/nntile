/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/autograd/function.hh
 * Base class for autograd functions
 *
 * @version 1.1.0
 * */

#pragma once

//#include <nntile/tensor.hh>
#include <memory>
#include <vector>
#include <string>

namespace nntile::autograd
{

//! Tensor handle type (mock for now)
class TensorHandle: public std::shared_ptr<float>
{
public:
    bool requires_grad_;

    TensorHandle(bool requires_grad=true):
        requires_grad_(requires_grad)
    {
    }

    bool requires_grad() const { return requires_grad_; }
    void set_requires_grad(bool requires_grad) { requires_grad_ = requires_grad; }
};

//! Base class for all autograd functions
class Function
{
protected:
    //! Saved tensors for backward pass
    std::vector<TensorHandle> saved_tensors_;

    //! Whether the function requires gradient computation
    bool requires_grad_;

    //! Function name for debugging
    std::string name_;

public:
    //! Constructor
    Function(
        const std::string& name,
        bool requires_grad=true
    ):
        requires_grad_(requires_grad),
        name_(name)
    {
        saved_tensors_.clear();
    }

    //! Virtual destructor
    virtual ~Function() = default;

    //! Forward pass implementation
    virtual std::vector<TensorHandle> forward(
        const std::vector<TensorHandle>& inputs) = 0;

    //! Backward pass implementation
    virtual std::vector<TensorHandle> backward(
        const std::vector<TensorHandle>& grad_outputs) = 0;

    //! Save tensors for backward pass
    void save_for_backward(const std::vector<TensorHandle>& tensors)
    {
        saved_tensors_ = tensors;
    }

    //! Check if function requires gradient computation
    bool requires_grad() const { return requires_grad_; }

    //! Get function name
    const std::string& name() const { return name_; }

    //! Get saved tensors
    const std::vector<TensorHandle>& saved_tensors() const { return saved_tensors_; }
};

} // namespace nntile::autograd
