#pragma once

#include "nntile/tile/tile.hh"

using nntile::Tile;

template<typename T>
int check_tiles_intersection(const Tile<T> &A, const Tile<T> &B)
{
    size_t ndim = A.ndim;
    A.acquire(STARPU_R);
    B.acquire(STARPU_R);
    const auto A_ptr = A.get_local_ptr(), B_ptr = B.get_local_ptr();
    int ok = 1;
    for(size_t i = 0; i < A.nelems; ++i)
    {
        auto index = A.linear_to_index(i);
        if(B.contains_index(index))
        {
            size_t B_linear_offset = B.index_to_linear(index);
            if(A_ptr[i] != B_ptr[B_linear_offset])
            {
                ok = 0;
                break;
            }
        }
    }
    A.release();
    B.release();
    return ok;
}

template<typename T>
int check_tiles_intersection(const Tile<T> &A,
        const std::vector<size_t> &A_offset, const Tile<T> &B,
        const std::vector<size_t> &B_offset)
{
    Tile<T> A2(A), B2(B);
    A2.offset = A_offset;
    B2.offset = B_offset;
    return check_tiles_intersection(A2, B2);
}

