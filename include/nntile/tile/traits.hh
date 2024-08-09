/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/traits.hh
 * Integer properties of the Tile<T> class
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <vector>
#include <array>
#include <stdexcept>
#include <iostream>

namespace nntile::tile
{

//! Integer arithmetics for tiles, that are arrays stored contiguously
class TileTraits
{
    //! Check if number of dimensions is not too much
    static Index _check_ndim(const std::vector<Index> &shape)
    {
        // Check if number of dimensions fits into Index type
        Index ndim = shape.size();
        if(shape.size() != ndim)
        {
            throw std::runtime_error("Number of dimensions does not fit "
                    "Index type");
        }
        return ndim;
    }
    //! Check is shape is positive
    static std::vector<Index> _check_shape(const std::vector<Index> &shape)
    {
        // Dimensions of inputs are already checked to match
        Index ndim = shape.size();
        for(Index i = 0; i < ndim; ++i)
        {
            if(shape[i] <= 0)
            {
                throw std::runtime_error("shape[i] <= 0");
            }
        }
        return shape;
    }
public:
    //! Dimensionality.
    Index ndim;
    //! Shape of the tile.
    std::vector<Index> shape;
    //! Stride of the tile.
    /*! stride[0] = 1, while stride[i+1] = stride[i] * shape[i].
     * */
    std::vector<Index> stride;
    //! Number of elements in the tile
    Index nelems;
    //! Shapes of all possible reshapes into matrices
    /*! matrix_shape[0] is a (prod(shape[0:0]), prod(shape[0:ndim]) reshape
     * matrix_shape[1] is a (prod(shape[0:1]), prod(shape[1:ndim]) reshape
     * and so on, matrix_shape[ndim] is a (prod(shape[0:ndim]),
     * prod(shape[ndim:ndim]) reshape
     * */
    std::vector<std::array<Index, 2>> matrix_shape;
    //! Construct a integer properties of a tile
    /*! @param[in] shape_: Shape of the tile itself
     * */
    explicit TileTraits(const std::vector<Index> &shape_):
        // Check if number of dimensions fits into Index type
        ndim(_check_ndim(shape_)),
        // Check input shape
        shape(_check_shape(shape_)),
        // No other checks are required, just allocate space
        stride(ndim),
        matrix_shape(ndim+1)
    {
        // Compute number of rows of reshapes into matrices
        Index tmp = 1;
        matrix_shape[0][0] = 1;
        for(Index i = 1; i <= ndim; ++i)
        {
            tmp *= shape[i-1];
            matrix_shape[i][0] = tmp;
        }
        // Set total number of elements in a tile
        nelems = tmp;
        // Compute number of columns of reshapes into matrices
        tmp = 1;
        matrix_shape[ndim][1] = 1;
        for(Index i = ndim; i >= 1; --i)
        {
            tmp *= shape[i-1];
            matrix_shape[i-1][1] = tmp;
        }
        // Set tile stride
        for(Index i = 0; i < ndim; ++i)
        {
            stride[i] = matrix_shape[i][0];
        }
    }
    //! Linear memory offset of a tile element with proper bounds check.
    /*! This function shall be used for debugging and testing
     *
     * @param[in] index: Coordinate of an element. Shall belong to the tile,
     *      i.e. 0 <= index[i] < shape[i].
     * @returns Offset of the element in a corresponding linear memory,
     *      assuming Fortran-order storage of tile elements.
     * */
    Index index_to_linear(const std::vector<Index> &index) const
    {
        // Check number of dimensions
        if(index.size() != ndim)
        {
            throw std::runtime_error("Wrong dimensionality");
        }
        // In case of a scalar tile there is only 1 element
        if(ndim == 0)
        {
            return 0;
        }
        // Check tile bounds
        if(index[0] < 0 or index[0] >= shape[0])
        {
            throw std::runtime_error("Index out of bounds");
        }
        // Get the actual memory offset
        Index linear_offset = index[0]; // stride[0]=1
        for(Index i = 1; i < ndim; ++i)
        {
            if(index[i] < 0 or index[i] >= shape[i])
            {
                throw std::runtime_error("Index out of bounds");
            }
            linear_offset += index[i] * stride[i];
        }
        return linear_offset;
    }
    //! Coordinate of a tile element with proper bounds check.
    /*! This function shall be used for debugging and testing
     *
     * @param[in] linear_offset: Linear memory offset in range [0,nelems)
     * @returns Coordinate of the corresponding element,
     *      assuming Fortran-order storage of tile elements.
     * */
    std::vector<Index> linear_to_index(Index linear_offset) const
    {
        // Check bounds
        if(linear_offset < 0 or linear_offset >= nelems)
        {
            throw std::runtime_error("Index out of bounds");
        }
        // Scalar case
        if(ndim == 0)
        {
            return std::vector<Index>();
        }
        // Other cases
        std::vector<Index> index(ndim);
        for(Index i = ndim-1; i >= 1; --i)
        {
            const Index div = linear_offset / stride[i];
            linear_offset -= div * stride[i];
            index[i] = div;
        }
        index[0] = linear_offset;
        return index;
    }
    //! Check if tile contains given coordinate
    bool contains_index(const std::vector<Index> &index) const
    {
        if(index.size() != ndim)
        {
            throw std::runtime_error("Invalid dimensionality");
        }
        for(Index i = 0; i < ndim; ++i)
        {
            if(index[i] < 0 or index[i] >= shape[i])
            {
                return false;
            }
        }
        return true;
    }
    // Output information about tile properties
    friend std::ostream &operator<<(std::ostream &os,
            const TileTraits &traits);
};

} // namespace nntile::tile
