#pragma once

#include <array>
#include <vector>
#include <stdexcept>
#include <iostream>

namespace nntile
{

namespace _Trans
{
    constexpr struct NoTrans
    {
    } NoTrans;
    constexpr struct Trans
    {
    } Trans;
};

struct TransOp
{
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
    constexpr TransOp(const struct _Trans::NoTrans):
        value(NoTrans)
    {
    }
    constexpr TransOp(const struct _Trans::Trans):
        value(Trans)
    {
    }
    template<typename T>
    TransOp(const T &) = delete;
    template<typename T>
    operator T() = delete;
};

namespace Debug
{
    constexpr struct Debug
    {
    } Debug;

    constexpr struct NoDebug
    {
    } NoDebug;
}

struct ContiguousTileTraits
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
    ContiguousTileTraits(const std::vector<int> &shape_):
        ndim(shape_.size()),
        shape(shape_),
        stride(ndim),
        matrix_shape(ndim-1)
    {
        // Check if ndim is non-zero
        if(ndim <= 0)
        {
            throw std::runtime_error("shape must be non-empty");
        }
        // Check if input shape is positive
        for(size_t i = 0; i < ndim; ++i)
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
        for(size_t i = 1; i < ndim-1; ++i)
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
        for(size_t i = 0; i < ndim-1; ++i)
        {
            stride[i+1] = matrix_shape[i][0];
        }
    }
    template<size_t NDIM>
    ContiguousTileTraits(const std::array<int, NDIM> &shape_):
        ContiguousTileTraits(std::vector<int>(shape_.cbegin(), shape_.cend()))
    {
        static_assert(NDIM != 0);
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
        for(size_t i = 1; i < ndim; ++i)
        {
            if((index[i] < 0) or (index[i] >= shape[i]))
            {
                throw std::runtime_error("Index out of bounds");
            }
            offset += index[i] * stride[i];
        }
        return offset;
    }
    template<size_t NDIM>
    int offset(const std::array<int, NDIM> &index) const
    {
        if(NDIM != ndim)
        {
            throw std::runtime_error("Wrong dimensionality");
        }
        if((index[0] < 0) or (index[0] >= shape[0]))
        {
            throw std::runtime_error("Index out of bounds");
        }
        int offset = index[0]; // stride[0]=1
        for(size_t i = 1; i < NDIM; ++i)
        {
            if((index[i] < 0) or (index[i] >= shape[i]))
            {
                throw std::runtime_error("Index out of bounds");
            }
            offset += index[i] * stride[i];
        }
        return offset;
    }
};

//! Check if dimensionalities of tensors match gemm
inline constexpr void gemm_check_ndim(const ContiguousTileTraits &A,
        const ContiguousTileTraits &B, const ContiguousTileTraits &C,
        int ndim=1)
{
    if(ndim <= 0)
    {
        throw std::runtime_error("ndim <= 0");
    }
    if(A.ndim < ndim)
    {
        throw std::runtime_error("A.ndim < ndim");
    }
    if(B.ndim < ndim)
    {
        throw std::runtime_error("B.ndim < ndim");
    }
    if(C.ndim < ndim)
    {
        throw std::runtime_error("C.ndim < ndim");
    }
    if(A.ndim + B.ndim - C.ndim != 2*ndim)
    {
        throw std::runtime_error("A.ndim + B.ndim - C.ndim != 2*ndim");
    }
}

//! Check if shapes of tensors A and B match gemm
inline void gemm_check_AB(const struct _Trans::NoTrans &,
        const ContiguousTileTraits &A, const struct _Trans::NoTrans &,
        const ContiguousTileTraits &B, int ndim=1)
{
    for(int i = 0; i < ndim; ++i)
    {
        if(A.shape[A.ndim-ndim+i] != B.shape[i])
        {
            throw std::runtime_error("A.shape[A.ndim-ndim:A.ndim-1] != "
                    "B.shape[0:ndim-1]");
        }
    }
}

//! Check if shapes of tensors A and B match gemm
inline void gemm_check_AB(const struct _Trans::Trans &,
        const ContiguousTileTraits &A, const struct _Trans::NoTrans &,
        const ContiguousTileTraits &B, int ndim=1)
{
    for(int i = 0; i < ndim; ++i)
    {
        if(A.shape[ndim-1-i] != B.shape[i])
        {
            throw std::runtime_error("A.shape[ndim-1-i] != B.shape[i]");
        }
    }
}

//! Check if shapes of tensors A and B match gemm
inline void gemm_check_AB(const struct _Trans::NoTrans &,
        const ContiguousTileTraits &A, const struct _Trans::Trans &,
        const ContiguousTileTraits &B, int ndim=1)
{
    for(int i = 0; i < ndim; ++i)
    {
        if(A.shape[A.ndim-ndim+i] != B.shape[B.ndim-1-i])
        {
            throw std::runtime_error("A.shape[A.ndim-ndim+i] != "
                    "B.shape[B.ndim-1-i]");
        }
    }
}

//! Check if shapes of tensors A and B match gemm
inline void gemm_check_AB(const struct _Trans::Trans &,
        const ContiguousTileTraits &A, const struct _Trans::Trans &,
        const ContiguousTileTraits &B, int ndim=1)
{
    for(int i = 0; i < ndim; ++i)
    {
        if(A.shape[ndim-1-i] != B.shape[B.ndim-1-i])
        {
            throw std::runtime_error("A.shape[0] != B.shape[B.ndim-1]");
        }
    }
}

//! Check if shapes of tensors A and B match gemm
inline constexpr void gemm_check_AB(const TransOp &transA,
        const ContiguousTileTraits &A, const TransOp &transB,
        const ContiguousTileTraits &B, int ndim=1)
{
    switch(transB.value)
    {
        case TransOp::NoTrans:
            switch(transA.value)
            {
                case TransOp::NoTrans:
                    gemm_check_AB(_Trans::NoTrans, A, _Trans::NoTrans, B,
                            ndim);
                    break;
                case TransOp::Trans:
                    gemm_check_AB(_Trans::Trans, A, _Trans::NoTrans, B, ndim);
                    break;
                default:
                    throw std::runtime_error("Wrong value of transA");
            }
            break;
        case TransOp::Trans:
            switch(transA.value)
            {
                case TransOp::NoTrans:
                    gemm_check_AB(_Trans::NoTrans, A, _Trans::Trans, B, ndim);
                    break;
                case TransOp::Trans:
                    gemm_check_AB(_Trans::Trans, A, _Trans::Trans, B, ndim);
                    break;
                default:
                    throw std::runtime_error("Wrong value of transA");
            }
            break;
        default:
            throw std::runtime_error("Wrong value of transB");
    }
}

//! Check if shapes of tensors A and C match gemm
inline void gemm_check_AC(const struct _Trans::NoTrans &,
        const ContiguousTileTraits &A, const ContiguousTileTraits &C,
        int ndim=1)
{
    for(int i = 0; i < A.ndim-ndim; ++i)
    {
        if(A.shape[i] != C.shape[i])
        {
            throw std::runtime_error("A.shape[i] != C.shape[i]");
        }
    }
}

//! Check if shapes of tensors A and C match gemm
inline void gemm_check_AC(const struct _Trans::Trans &,
        const ContiguousTileTraits &A, const ContiguousTileTraits &C,
        int ndim=1)
{
    for(int i = A.ndim-1, j = 0; i >= ndim; --i, ++j)
    {
        if(A.shape[i] != C.shape[j])
        {
            throw std::runtime_error("A.shape[i] != C.shape[j]");
        }
    }
}

//! Check if shapes of tensors A and C match gemm
inline void gemm_check_AC(const struct TransOp &transA,
        const ContiguousTileTraits &A, const ContiguousTileTraits &C,
        int ndim=1)
{
    switch(transA.value)
    {
        case TransOp::NoTrans:
            gemm_check_AC(_Trans::NoTrans, A, C, ndim);
            break;
        case TransOp::Trans:
            gemm_check_AC(_Trans::Trans, A, C, ndim);
            break;
        default:
            throw std::runtime_error("Wrong value of transA");
    }
}

//! Check if shapes of tensors B and C match gemm
inline void gemm_check_BC(const struct _Trans::NoTrans &,
        const ContiguousTileTraits &B, const ContiguousTileTraits &C,
        int ndim=1)
{
    for(size_t i = ndim, j = C.ndim-B.ndim+ndim; i < B.ndim; ++i, ++j)
    {
        if(B.shape[i] != C.shape[j])
        {
            throw std::runtime_error("B.shape[i] != C.shape[j]");
        }
    }
}

//! Check if shapes of tensors B and C match gemm
inline void gemm_check_BC(const struct _Trans::Trans &,
        const ContiguousTileTraits &B, const ContiguousTileTraits &C,
        int ndim=1)
{
    for(size_t i = B.ndim-1-ndim, j = C.ndim-B.ndim+ndim; i > 0; --i, ++j)
    {
        if(B.shape[i] != C.shape[j])
        {
            throw std::runtime_error("B.shape[i] != C.shape[j]");
        }
    }
}

//! Check if shapes of tensors A and C match gemm
inline void gemm_check_BC(const struct TransOp &transB,
        const ContiguousTileTraits &B, const ContiguousTileTraits &C,
        int ndim=1)
{
    switch(transB.value)
    {
        case TransOp::NoTrans:
            gemm_check_BC(_Trans::NoTrans, B, C, ndim);
            break;
        case TransOp::Trans:
            gemm_check_BC(_Trans::Trans, B, C, ndim);
            break;
        default:
            throw std::runtime_error("Wrong value of transB");
    }
}

//! Check if tensors match gemm
template<typename TA, typename TB>
void gemm_check(const TA &transA, const ContiguousTileTraits &A,
        const TB &transB, const ContiguousTileTraits &B,
        ContiguousTileTraits &C, int ndim=1)
{
    // Check if dimensionalities match
    gemm_check_ndim(A, B, C, ndim);
    // Check if shapes of A and B match
    gemm_check_AB(transA, A, transB, B, ndim);
    // Check if shapes of A and C match
    gemm_check_AC(transA, A, C, ndim);
    // Check if shapes of B and C match
    gemm_check_BC(transB, B, C, ndim);
}

} // namespace nntile

