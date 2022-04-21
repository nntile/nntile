#pragma once

#include "nntile/tensor/tensor.hh"
#include "../tile/check_tiles_intersection.hh"

using nntile::Tensor;

template<typename T>
bool check_tensors_intersection(const Tensor<T> &A,
        const std::vector<Index> &A_offset, const Tensor<T> &B,
        const std::vector<Index> &B_offset)
{
    if(A.ndim != A_offset.size())
    {
        throw std::runtime_error("A.ndim != A_offset.size()");
    }
    if(B.ndim != A.ndim)
    {
        throw std::runtime_error("B.ndim != A.ndim");
    }
    if(B_offset.size() != B.ndim)
    {
        throw std::runtime_error("B_offset.size() != B.ndim");
    }
    for(Index i = 0; i < A.grid.nelems; ++i)
    {
        auto A_tile = A.get_tile(i);
        auto A_tile_index = A.grid.linear_to_index(i);
        auto A_tile_offset(A_offset);
        for(Index j = 0; j < A.ndim; ++j)
        {
            A_tile_offset[j] += A_tile_index[j] * A.basetile_shape[j];
        }
        for(Index j = 0; j < B.grid.nelems; ++j)
        {
            auto B_tile = B.get_tile(j);
            auto B_tile_index = B.grid.linear_to_index(j);
            auto B_tile_offset(B_offset);
            for(Index k = 0; k < A.ndim; ++k)
            {
                B_tile_offset[k] += B_tile_index[k] * B.basetile_shape[k];
            }
            if(!check_tiles_intersection(A_tile, A_tile_offset, B_tile,
                    B_tile_offset))
            {
                return false;
            }
        }
    }
    return true;
}

template<typename T>
bool check_tensors_intersection(const Tensor<T> &A, const Tensor<T> &B)
{
    return check_tensors_intersection(A, std::vector<Index>(A.ndim),
            B, std::vector<Index>(B.ndim));
}

