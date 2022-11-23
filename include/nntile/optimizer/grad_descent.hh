/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/optimizer/grad_descent.hh
 * Gradient descent with constant learning rate
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-11-23
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
class GradDescent: public Base<T>
{
public:
    tensor::Tensor<T> lr;
    GradDescent(const std::vector<tensor::Tensor<T>> &params_,
            const std::vector<tensor::Tensor<T>> &grads_,
            T learning_rate_, starpu_mpi_tag_t &last_tag):
        Base<T>(params_, grads_),
        lr(tensor::TensorTraits({}, {}), {0}, last_tag)
    {
        // MPI root sets the learning rate
        if(starpu_mpi_world_rank() == 0)
        {
            auto lr_tile = lr.get_tile(0).acquire(STARPU_W);
            lr_tile[0] = -learning_rate_;
            lr_tile.release();
        }
    }
    void update()
    {
        for(Index i = 0; i < Base<T>::params.size(); ++i)
        {
            tensor::axpy_async(lr, Base<T>::grads[i], Base<T>::params[i]);
        }
    }
};

// Explicit instantiation
extern template
class GradDescent<fp32_t>;

extern template
class GradDescent<fp64_t>;

} // namespace optimizer
} // namespace nntile

