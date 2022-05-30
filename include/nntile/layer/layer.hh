/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/layer/layer.hh
 * Base class for all layers
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#pragma once

namespace nntile
{

class Layer
{
public:
    virtual forward(const Tensor<T> &) = delete;
    virtual backward(const Tensor<T> &) = delete;
};

}

