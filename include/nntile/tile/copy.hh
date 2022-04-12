#pragma once

#include <nntile/tile/tile.hh>

namespace nntile
{

template<typename T>
void copy_async(const Tile<T> &src, const std::vector<size_t> &src_coord,
        const Tile<T> &dst, const std::vector<size_t> &dst_coord);

extern template
void copy_async(const Tile<float> &src, const std::vector<size_t> &src_coord,
        const Tile<float> &dst, const std::vector<size_t> &dst_coord);

extern template
void copy_async(const Tile<double> &src, const std::vector<size_t> &src_coord,
        const Tile<double> &dst, const std::vector<size_t> &dst_coord);

template<typename T>
void copy_async(const Tile<T> &src, const Tile<T> &dst)
{
    copy_async<T>(src, std::vector<size_t>(src.ndim, 0), dst,
            std::vector<size_t>(dst.ndim, 0));
}

template<typename T>
void copy(const Tile<T> &src, const std::vector<size_t> &src_coord,
        const Tile<T> &dst, const std::vector<size_t> &dst_coord)
{
    copy_async<T>(src, src_coord, dst, dst_coord);
    starpu_task_wait_for_all();
}

template<typename T>
void copy(const Tile<T> &src, const Tile<T> &dst)
{
    copy_async<T>(src, std::vector<size_t>(src.ndim, 0), dst,
            std::vector<size_t>(dst.ndim, 0));
    starpu_task_wait_for_all();
}

} // namespace nntile

