/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/autograd/add.hh
 * Add function for autograd
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/autograd/function.hh>
#include <nntile/autograd/tensor.hh>

namespace nntile::autograd
{

//! Add function for autograd
class Add: public Function
{
private:
    //! Alpha multiplier for first input
    Scalar alpha_;
    //! Beta multiplier for second input
    Scalar beta_;

public:
    //! Constructor
    Add(Scalar alpha, Scalar beta, bool requires_grad = true):
        Function("add", requires_grad),
        alpha_(alpha),
        beta_(beta)
    {
    }

    //! Forward pass implementation
    std::vector<TensorHandle> forward(
        const std::vector<TensorHandle>& inputs)
        override;

    //! Backward pass implementation
    std::vector<TensorHandle> backward(
        const std::vector<TensorHandle>& grad_outputs)
        override;
};

//! Create an AutogradTensor that represents addition of two tensors
AutogradTensor add(
    Scalar alpha,
    const AutogradTensor& a,
    Scalar beta,
    const AutogradTensor& b
);

} // namespace nntile::autograd
