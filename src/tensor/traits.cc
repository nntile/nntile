/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/traits.cc
 * Integer properties of the Tensor<T> class
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/traits.hh"

namespace nntile::tensor
{

std::ostream &operator<<(std::ostream &os, const TensorTraits &traits)
{
    os << "TensorTraits object at " << &traits << "\n";
    os << "ndim=" << traits.ndim << "\n";
    os << "shape=(";
    if(traits.ndim > 0)
    {
        os << traits.shape[0];
        for(Index i = 1; i < traits.ndim; ++i)
        {
            os << "," << traits.shape[i];
        }
    }
    os << ")\n";
    os << "basetile_shape=(";
    if(traits.ndim > 0)
    {
        os << traits.basetile_shape[0];
        for(Index i = 1; i < traits.ndim; ++i)
        {
            os << "," << traits.basetile_shape[i];
        }
    }
    os << ")\n";
    os << "leftover_shape=(";
    if(traits.ndim > 0)
    {
        os << traits.leftover_shape[0];
        for(Index i = 1; i < traits.ndim; ++i)
        {
            os << "," << traits.leftover_shape[i];
        }
    }
    os << ")\n";
    os << "grid\n" << traits.grid;
    os << "Tiles\n";
    for(Index i = 0; i < traits.grid.nelems; ++i)
    {
        const auto index = traits.grid.linear_to_index(i);
        os << "  " << i << "\n";
        os << "    index=(";
        if(traits.ndim > 0)
        {
            os << index[0];
            for(Index j = 1; j < traits.ndim; ++j)
            {
                os << "," << index[j];
            }
        }
        os << ")\n";
        const auto shape = traits.get_tile_shape(index);
        os << "    shape=(";
        if(traits.ndim > 0)
        {
            os << shape[0];
            for(Index j = 1; j < traits.ndim; ++j)
            {
                os << "," << shape[j];
            }
        }
        os << ")\n";
    }
    return os;
}

} // namespace nntile::tensor
