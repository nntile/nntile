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
    //! Check if dimensionalities of inputs match
    static size_t _check_ndim(
            const std::vector<size_t> &shape,
            const std::vector<size_t> &offset,
            const std::vector<size_t> &underlying_shape)
    {
        size_t ndim = shape.size();
        if(offset.size() != ndim)
        {
            throw std::runtime_error("offset.size() != ndim");
        }
        if(underlying_shape.size() != ndim)
        {
            throw std::runtime_error("underlying_shape.size() != ndim");
        }
        return ndim;
    }
    //! Check shape and if tile is fully inside underlying tensor
    static std::vector<size_t> _check_shape(
            const std::vector<size_t> &shape,
            const std::vector<size_t> &offset,
            const std::vector<size_t> &underlying_shape)
    {
        // Dimensions of inputs are already checked to match
        size_t ndim = shape.size();
        for(size_t i = 0; i < ndim; ++i)
        {
            if(shape[i] == 0)
            {
                throw std::runtime_error("shape[i] == 0");
            }
            if(shape[i]+offset[i] > underlying_shape[i])
            {
                throw std::runtime_error("shape[i]+offset[i] > "
                        "underlying_shape[i]");
            }
        }
        return shape;
    }
public:
    //! Dimensionality.
    size_t ndim;
    //! Shape of the tile.
    std::vector<size_t> shape;
    //! Stride of the tile.
    //
    // stride[0] = 1, while stride[i+1] = stride[i] * shape[i].
    std::vector<size_t> stride;
    //! Number of elements in the tile
    size_t nelems;
    //! Shapes of all possible reshapes into matrices
    //
    // matrix_shape[0] is a (prod(shape[0:0]), prod(shape[0:ndim]) reshape
    // matrix_shape[1] is a (prod(shape[0:1]), prod(shape[1:ndim]) reshape
    // and so on, matrix_shape[ndim] is a (prod(shape[0:ndim]),
    // prod(shape[ndim:ndim]) reshape
    std::vector<std::array<size_t, 2>> matrix_shape;
    //! Offset of the tile in the underlying tensor
    std::vector<size_t> offset;
    //! Shape of the underlying tensor containing this tile
    std::vector<size_t> underlying_shape;
    //! Stride of the underlying tensor
    std::vector<size_t> underlying_stride;
    //! Construct a tile that is a part of underlying tensor
    //
    // @param[in] shape_: Shape of the tile itself
    // @param[in] offset_: Offset of the tile in the underlying tensor
    // @param[in] underlying_shape_: Shape of the underlying tensor
    TileTraits(const std::vector<size_t> &shape_,
            const std::vector<size_t> &offset_,
            const std::vector<size_t> &underlying_shape_):
        // Check dimensions of input shapes
        ndim(_check_ndim(shape_, offset_, underlying_shape_)),
        // Check input shapes
        shape(_check_shape(shape_, offset_, underlying_shape_)),
        // No other checks are required
        stride(ndim),
        matrix_shape(ndim+1),
        offset(offset_),
        underlying_shape(underlying_shape_),
        underlying_stride(ndim)
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
        // Set stride for the underlying tensor
        tmp = 1;
        for(size_t i = 0; i < ndim; ++i)
        {
            underlying_stride[i] = tmp;
            tmp *= underlying_shape[i];
        }
    }
    //! Construct a tile that is the underlying tensor itself
    //
    // @param[in] shape_: Shape of the tile and the underlying tensor
    TileTraits(const std::vector<size_t> &shape_):
        TileTraits(shape_, std::vector<size_t>(shape_.size()), shape_)
    {
    }
    //! Linear memory offset of a tensor element with proper bounds check.
    //
    // @param[in] index: Coordinate of an element. Shall belong to the tile,
    //      i.e. offset[i] <= index[i] < offset[i]+shape[i].
    // @returns Offset of the element in a corresponding linear memory,
    //      assuming Fortran-order storage of the tile elements.
    size_t index_to_linear(const std::vector<size_t> &index) const
    {
        if(index.size() != ndim)
        {
            throw std::runtime_error("Wrong dimensionality");
        }
        // In case of a scalar tile/tensor there is only 1 element
        if(ndim == 0)
        {
            return 0;
        }
        // Check tile bounds
        if(index[0] < offset[0] or index[0] >= shape[0]+offset[0])
        {
            throw std::runtime_error("Index out of bounds");
        }
        // Get the actual memory offset
        size_t linear_offset = index[0] - offset[0]; // stride[0]=1
        for(size_t i = 1; i < ndim; ++i)
        {
            if(index[i] < offset[i] or index[i] >= shape[i]+offset[i])
            {
                throw std::runtime_error("Index out of bounds");
            }
            linear_offset += (index[i]-offset[i]) * stride[i];
        }
        return linear_offset;
    }
    //! Coordinate of a tensor element with proper bounds check.
    //
    // @param[in] linear: Linear memory offset in range [0,nelems)
    // @returns Coordinate of the corresponding element,
    //      assuming Fortran-order storage of the tile elements.
    std::vector<size_t> linear_to_index(size_t linear_offset) const
    {
        // Check bounds
        if(linear_offset >= nelems)
        {
            throw std::runtime_error("Index out of bounds");
        }
        // Scalar case
        if(ndim == 0)
        {
            return std::vector<size_t>();
        }
        // Other cases
        std::vector<size_t> index(ndim);
        // Avoid i=0-1 in size_t type
        for(size_t i = ndim-1; i >= 1; --i)
        {
            const size_t div = linear_offset / stride[i];
            linear_offset -= div * stride[i];
            index[i] = div + offset[i];
        }
        index[0] = linear_offset + offset[0];
        return index;
    }
    //! Check if tile contains given coordinate
    int contains_index(const std::vector<size_t> &index) const noexcept
    {
        for(size_t i = 0; i < ndim; ++i)
        {
            if(index[i] < offset[i] or index[i] >= offset[i]+shape[i])
            {
                return 0;
            }
        }
        return 1;
    }
    friend std::ostream &operator<<(std::ostream &os,
            const TileTraits &traits);
};

} // namespace nntile

