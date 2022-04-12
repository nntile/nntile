#pragma once

#include "nntile/tile/tile.hh"

using nntile::Tile;

template<typename T>
void check_tiles_intersection(const Tile<T> &src,
        const std::vector<size_t> &src_coord, const Tile<T> &dst,
        const std::vector<size_t> &dst_coord)
{
    size_t ndim = src.ndim;
    const auto src_ptr = src.get_local_ptr(), dst_ptr = dst.get_local_ptr();
    std::vector<size_t> src_index(ndim), dst_index(ndim, 0);
    bool ignore_element = false;
    for(size_t k = 0; k < ndim; ++k)
    {
        size_t global_coord = dst_coord[k];
        if((global_coord >= src_coord[k]+src.shape[k])
                or (global_coord < src_coord[k]))
        {
            ignore_element = true;
            break;
        }
        src_index[k] = global_coord - src_coord[k];
    }
    if(!ignore_element)
    {
        size_t src_offset = src_index[0];
        for(size_t k = 1; k < ndim; ++k)
        {
            src_offset += src_index[k] * src.stride[k];
        }
        if(dst_ptr[0] != src_ptr[src_offset])
        {
            throw std::runtime_error("dst_ptr[0] != src_ptr[src_offset]");
        }
    }
    for(size_t i = 1; i < dst.nelems; ++i)
    {
        ++dst_index[0];
        size_t j = 0;
        while(dst_index[j] == dst.shape[j])
        {
            dst_index[j] = 0;
            ++j;
            ++dst_index[j];
        }
        ignore_element = false;
        for(size_t k = 0; k < ndim; ++k)
        {
            size_t global_coord = dst_index[k] + dst_coord[k];
            if((global_coord >= src_coord[k]+src.shape[k])
                    or (global_coord < src_coord[k]))
            {
                ignore_element = true;
                break;
            }
            src_index[k] = global_coord - src_coord[k];
        }
        if(!ignore_element)
        {
            size_t src_offset = src_index[0];
            for(size_t k = 1; k < ndim; ++k)
            {
                src_offset += src_index[k] * src.stride[k];
            }
            if(dst_ptr[i] != src_ptr[src_offset])
            {
                throw std::runtime_error("dst_ptr[i] != src_ptr[src_offset]");
            }
        }
    }
}

