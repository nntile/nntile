/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/optimizer/sgd.hh
 * Stochastic Gradient Descent with constant learning rate.
 * It supports weight decay, momentum and Nesterov regimes. 
 * The formulas are similar to PyTorch SGD
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @date 2023-02-10
 * */

#pragma once

#include <nntile/optimizer/base.hh>
#include <nntile/tensor/axpy.hh>

namespace nntile
{
namespace optimizer
{

//! Common API for all optimizers
template<typename T>
class SGD: public Base<T>
{
public:
    T lr;
    T momentum;
    bool nesterov;
    T damping;
    T weight_decay;
    Index num_iter = Index(0.);
    std::vector<tensor::Tensor<T>> states;
    SGD(const std::vector<tensor::Tensor<T>> &params_,
            const std::vector<tensor::Tensor<T>> &grads_,
            T learning_rate_, starpu_mpi_tag_t &last_tag,
            T momentum_ = T(0.), T damping_ = T(0.),
            bool nesterov_=false, T weight_decay_ = T(0.)):
        Base<T>(params_, grads_),
        lr(-learning_rate_),
        momentum(momentum_),
        damping(damping_),
        weight_decay(weight_decay_)
    {
        if (momentum > 0) {
            for (Index i = 0; i < Base<T>::params.size(); ++i) {
                states[i](Base<T>::params[i], 0, last_tag);
                states[i].clear();
            }
        }
    }

    void update()
    {
        for (Index i = 0; i < Base<T>::params.size(); ++i)
        {
            if (weight_decay != T(0.)) {
                tensor::axpy2_async(weight_decay, Base<T>::params[i], Base<T>::grads[i]);
            }
            if (momentum > 0) {
                if (num_iter == 0) {
                    tensor::copy(Base<T>::grads[i], states[i]);
                } else {
                    // FIX: multiplication of state tensor by scalar momentum is done via axpy operation
                    tensor::axpy2_async(momentum - 1, Base<T>::states[i], Base<T>::states[i]);
                    tensor::axpy2_async(1 - damping, Base<T>::grads[i], Base<T>::states[i]);
                }
                if (nesterov) {
                    tensor::axpy2_async(momentum, Base<T>::states[i], Base<T>::grads[i]);
                } else {
                    tensor::copy(ase<T>::states[i], Base<T>::grads[i]);
                }
            }
            tensor::axpy2_async(lr, Base<T>::grads[i], Base<T>::params[i]);
        }
        ++num_iter;
    }
};

// Explicit instantiation
extern template
class GradDescent<fp32_t>;

extern template
class GradDescent<fp64_t>;

} // namespace optimizer
} // namespace nntile

