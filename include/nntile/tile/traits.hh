#pragma once

#include <array>
#include <vector>
#include <stdexcept>
#include <iostream>

namespace nntile
{

class TransOp
{
public:
    enum Value: int
    {
        Undefined,
        NoTrans,
        Trans
    } value;
    constexpr TransOp():
        value(Undefined)
    {
    }
    constexpr TransOp(const enum Value &value_):
        value(value_)
    {
    }
    template<typename T>
    TransOp(const T &) = delete;
    template<typename T>
    operator T() = delete;
};

struct TileTraits
{
    //! Run-time dimensionality.
    int ndim;
    //! Shape of tile.
    std::vector<int> shape;
    //! Stride of tile.
    //
    // stride[0] = 1, while stride[i+1] = stride[i] * shape[i].
    std::vector<int> stride;
    //! Number of elements in tile, shall not exceed MAX_INT
    int nelems;
    //! Shapes of all possible reshapes into matrices
    //
    // matrix_shape[0] is a (prod(shape[0:0]), prod(shape[1:ndim-1]) reshape
    // matrix_shape[1] is a (prod(shape[0:1]), prod(shape[2:ndim-1]) reshape
    // and so on, matrix_shape[ndim-2] is a (prod(shape[0:ndim-2]),
    // prod(shape[ndim-1:ndim-1]) reshape
    std::vector<std::array<int, 2>> matrix_shape;
    //! Constructor
    TileTraits(const std::vector<int> &shape_):
        ndim(shape_.size()),
        shape(shape_),
        stride(ndim),
        matrix_shape(ndim-1)
    {
        // Check if ndim is non-zero
        if(ndim == 0)
        {
            throw std::runtime_error("shape must be non-empty");
        }
        // Double-check conversion from size_t to int
        if(ndim != shape.size())
        {
            throw std::runtime_error("size_t to int conversion overflow");
        }
        // Check if input shape is positive
        for(int i = 0; i < ndim; ++i)
        {
            if(shape[i] <= 0)
            {
                throw std::runtime_error("shape must be positive");
            }
        }
        // Use temporary in a long format to check for integer overflow
        size_t tmp_long = shape[0];
        matrix_shape[0][0] = shape[0];
        matrix_shape[ndim-2][1] = shape[ndim-1];
        for(int i = 1; i < ndim-1; ++i)
        {
            tmp_long *= shape[i];
            matrix_shape[i][0] = tmp_long;
            matrix_shape[ndim-2-i][1] = matrix_shape[ndim-1-i][1]
                * shape[ndim-1-i];
        }
        tmp_long *= shape[ndim-1];
        nelems = tmp_long;
        // Check for integer overflow
        if(nelems != tmp_long)
        {
            throw std::runtime_error("Integer overflow in tile arithmetic");
        }
        // Set other members
        stride[0] = 1;
        for(int i = 0; i < ndim-1; ++i)
        {
            stride[i+1] = matrix_shape[i][0];
        }
    }
    //! Offset from a starting point of tile to a given coordinate.
    int offset(const std::vector<int> &index) const
    {
        if(index.size() != ndim)
        {
            throw std::runtime_error("Wrong dimensionality");
        }
        if((index[0] < 0) or (index[0] >= shape[0]))
        {
            throw std::runtime_error("Index out of bounds");
        }
        int offset = index[0]; // stride[0]=1
        for(int i = 1; i < ndim; ++i)
        {
            if((index[i] < 0) or (index[i] >= shape[i]))
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

std::ostream &operator<<(std::ostream &os, const TileTraits &traits)
{
    os << "TileTraits object at " << &traits << "\n";
    os << "shape=(" << traits.shape[0];
    for(int i = 1; i < traits.ndim; ++i)
    {
        os << "," << traits.shape[i];
    }
    os << ")\n";
    os << "stride=(" << traits.stride[0];
    for(int i = 1; i < traits.ndim; ++i)
    {
        os << "," << traits.stride[i];
    }
    os << ")\n";
    os << "nelems=" << traits.nelems << "\n";
    os << "matrix_shape=((" << traits.matrix_shape[0][0] <<
        "," << traits.matrix_shape[0][1] << ")";
    for(int i = 1; i < traits.ndim-1; ++i)
    {
        os << ",(" << traits.matrix_shape[i][0] << "," <<
            traits.matrix_shape[i][1] << ")";
    }
    os << ")\n";
    return os;
}

} // namespace nntile

