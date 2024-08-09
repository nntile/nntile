/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/traits.cc
 * Integer properties of the Tile<T> class
 *
 * @version 1.1.0
 * */

#include "nntile/tile/traits.hh"

namespace nntile::tile
{

//! Output tile traits into stream
std::ostream &operator<<(std::ostream &os, const TileTraits &traits)
{
    // Output pointer and number of dimensions
    os << "TileTraits object at " << &traits << "\n";
    os << "ndim=" << traits.ndim << "\n";
    // Output shape
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
    // Output strides
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
    // Output number of elements
    os << "nelems=" << traits.nelems << "\n";
    // Output shapes of reshapes into different contiguous matrices
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

} // namespace nntile::tile
