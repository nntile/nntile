/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/traits.cc
 * Integer properties of the Tile<T> class
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#include "nntile/tile/traits.hh"

namespace nntile
{

std::ostream &operator<<(std::ostream &os, const TileTraits &traits)
{
    os << "TileTraits object at " << &traits << "\n";
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
    os << "stride=(";
    if(traits.ndim > 0)
    {
        os << traits.stride[0];
        for(Index i = 1; i < traits.ndim; ++i)
        {
            os << "," << traits.stride[i];
        }
    }
    os << ")\n";
    os << "nelems=" << traits.nelems << "\n";
    os << "matrix_shape=((" << traits.matrix_shape[0][0] <<
        "," << traits.matrix_shape[0][1] << ")";
    for(Index i = 1; i <= traits.ndim; ++i)
    {
        os << ",(" << traits.matrix_shape[i][0] << "," <<
            traits.matrix_shape[i][1] << ")";
    }
    os << ")\n";
    return os;
}

} // namespace nntile

