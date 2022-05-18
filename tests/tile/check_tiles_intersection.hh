#pragma once

#include "nntile/tile/tile.hh"

using nntile::Index;
using nntile::Tile;

template<typename T>
bool check_tiles_intersection(const Tile<T> &A,
        const std::vector<Index> &A_offset, const Tile<T> &B,
        const std::vector<Index> &B_offset)
{
    Index ndim = A.ndim;
    if(B.ndim != ndim)
    {
        throw std::runtime_error("B.ndim != ndim");
    }
    if(A_offset.size() != ndim)
    {
        throw std::runtime_error("A_offset.size() != ndim");
    }
    if(B_offset.size() != ndim)
    {
        throw std::runtime_error("B_offset.size() != ndim");
    }
    auto A_local = A.acquire(STARPU_R);
    auto B_local = B.acquire(STARPU_R);
    bool result = true;
    for(Index i = 0; i < A.nelems; ++i)
    {
        auto index = A.linear_to_index(i);
        for(Index j = 0; j < ndim; ++j)
        {
            index[j] += A_offset[j] - B_offset[j];
        }
        if(B.contains_index(index))
        {
            Index B_linear_offset = B.index_to_linear(index);
            if(A_local[i] != B_local[B_linear_offset])
            {
                std::cout << "A_local[" << i << "]=" << A_local[i] << "\n";
                std::cout << "B_local[" << B_linear_offset << "]=" <<
                    B_local[B_linear_offset] << "\n";
                result = false;
                break;
            }
        }
    }
    return result;
}

template<typename T>
int check_tiles_intersection(const Tile<T> &A, const Tile<T> &B)
{
    return check_tiles_intersection(A, std::vector<Index>(A.ndim),
            B, std::vector<Index>(B.ndim));
}

