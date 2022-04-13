#pragma once

#include <vector>
#include <array>
#include <stdexcept>
#include <iostream>

namespace nntile
{

//! Integer arithmetics for tiles
class TileTraits
{
public:
    //! Run-time dimensionality.
    size_t ndim;
    //! Shape of tile.
    std::vector<size_t> shape;
    //! Stride of tile.
    //
    // stride[0] = 1, while stride[i+1] = stride[i] * shape[i].
    std::vector<size_t> stride;
    //! Number of elements in tile, shall not exceed MAX_INT
    size_t nelems;
    //! Shapes of all possible reshapes into matrices
    //
    // matrix_shape[0] is a (prod(shape[0:0]), prod(shape[0:ndim]) reshape
    // matrix_shape[1] is a (prod(shape[0:1]), prod(shape[1:ndim]) reshape
    // and so on, matrix_shape[ndim] is a (prod(shape[0:ndim]),
    // prod(shape[ndim:ndim]) reshape
    std::vector<std::array<size_t, 2>> matrix_shape;
    //! Constructor
    TileTraits(const std::vector<size_t> &shape_):
        ndim(shape_.size()),
        shape(shape_),
        stride(ndim),
        matrix_shape(ndim+1)
    {
        // Compute number of rows of reshapes into matrices
        size_t tmp = 1;
        matrix_shape[0][0] = 1;
        for(size_t i = 1; i <= ndim; ++i)
        {
            tmp *= shape[i-1];
            matrix_shape[i][0] = tmp;
        }
        // Set total number of elements in a tile
        nelems = tmp;
        // Compute number of columns of reshapes into matrices
        tmp = 1;
        matrix_shape[ndim][1] = 1;
        // Avoid i=0-1 in size_t type
        for(size_t i = ndim; i >= 1; --i)
        {
            tmp *= shape[i-1];
            matrix_shape[i-1][1] = tmp;
        }
        // Set tile stride
        for(size_t i = 0; i < ndim; ++i)
        {
            stride[i] = matrix_shape[i][0];
        }
    }
    //! Offset from a starting point of tile to a given coordinate.
    size_t get_offset(const std::vector<size_t> &index) const
    {
        if(index.size() != ndim)
        {
            throw std::runtime_error("Wrong dimensionality");
        }
        if(ndim == 0)
        {
            return 0;
        }
        if(index[0] >= shape[0])
        {
            throw std::runtime_error("Index out of bounds");
        }
        size_t offset = index[0]; // stride[0]=1
        for(size_t i = 1; i < ndim; ++i)
        {
            if(index[i] >= shape[i])
            {
                throw std::runtime_error("Index out of bounds");
            }
            offset += index[i] * stride[i];
        }
        return offset;
    }
    friend std::ostream &operator<<(std::ostream &os,
            const TileTraits &traits);
};

} // namespace nntile

