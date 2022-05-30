/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/layer/fully_connected.hh
 * Fully connected layer
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#pragma once

#include <nntile/tensor.hh>

namespace nntile
{

template<typename T>
class FullyConnected
{
    Tensor<T> weight;
public:
    FullyConnected(const std::vector<Index> &shape,
            const std::vector<Index> &basetile):
        weight(shape, basetile)
    {
    }
    void init(const Tile<T> &tile)
    {
        if(tile.shape != weight.shape)
        {
            throw std::runtime_error("tile.shape != weight.shape");
        }
        copy_intersection(tile, weight);
    }
    void init(unsigned long long seed, T mean, T stddev)
    {
        randn(weight, seed, mean, stddev);
    }
    void forward_async(const Tensor<T> &input, const Tensor<T> &output)
    {
        constexpr T one = 1, zero = 0;
        gemm_async(one, TransOp::NoTrans, weight, TransOp::NoTrans, input,
                zero, output, 1);
    }
    const Tensor<T> &get_weight() const
    {
        return weight;
    }
    void unregister()
    {
        weight.unregister();
    }
};

}


