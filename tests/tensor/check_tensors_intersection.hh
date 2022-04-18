#pragma once

#include "nntile/tensor/tensor.hh"
#include "../tile/check_tiles_intersection.hh"

using nntile::Tensor;

template<typename T>
void check_tensors_intersection(const Tensor<T> &src,
        const std::vector<size_t> &src_coord, const Tensor<T> &dst,
        const std::vector<size_t> &dst_coord)
{
    for(size_t i = 0; i < src.grid.nelems; ++i)
    {
        auto src_tile = src.get_tile(i);
        auto src_index = src.get_tile_index(i);
        auto src_tile_coord(src_coord);
        std::cout << "src_tile_coord\n";
        for(size_t k = 0; k < src.ndim; ++k)
        {
            src_tile_coord[k] += src_index[k] * src.basetile_shape[k];
            std::cout << src_tile_coord[k] << " ";
        }
        std::cout << "\n";
        for(size_t j = 0; j < dst.grid.nelems; ++j)
        {
            auto dst_tile = dst.get_tile(j);
            auto dst_index = dst.get_tile_index(j);
            auto dst_tile_coord(dst_coord);
            std::cout << "dst_tile_coord\n";
            for(size_t k = 0; k < dst.ndim; ++k)
            {
                dst_tile_coord[k] += dst_index[k] * dst.basetile_shape[k];
                std::cout << dst_tile_coord[k] << " ";
            }
            std::cout << "\n";
            check_tiles_intersection(src_tile, src_tile_coord, dst_tile,
                    dst_tile_coord);
        }
    }
}

