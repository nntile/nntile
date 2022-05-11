/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/relu.cc
 * ReLU operation for Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#include "nntile/tensor/relu.hh"
#include "nntile/tile/relu.hh"

namespace nntile
{

template<typename T>
void relu_async(const Tensor<T> &A)
{
    for(Index i = 0; i < A.grid.nelems; ++i)
    {
        relu_async(A.get_tile(i));
    }
}

template
void relu_async(const Tensor<float> &A);

template
void relu_async(const Tensor<double> &A);

} // namespace nntile

