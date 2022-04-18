#pragma once

#include <nntile/tile/tile.hh>

namespace nntile
{

template<typename T>
void copy_intersection_async(const Tile<T> &src,
        const std::vector<size_t> &src_offset, const Tile<T> &dst,
        const std::vector<size_t> &dst_offset);

extern template
void copy_intersection_async(const Tile<float> &src,
        const std::vector<size_t> &src_offset, const Tile<float> &dst,
        const std::vector<size_t> &dst_offset);

extern template
void copy_intersection_async(const Tile<double> &src,
        const std::vector<size_t> &src_offset, const Tile<double> &dst,
        const std::vector<size_t> &dst_offset);

template<typename T>
void copy_intersection_async(const Tile<T> &src, const Tile<T> &dst)
{
    copy_intersection_async<T>(src, src.offset, dst, dst.offset);
}

template<typename T>
void copy_intersection(const Tile<T> &src,
        const std::vector<size_t> &src_offset, const Tile<T> &dst,
        const std::vector<size_t> &dst_offset)
{
    copy_intersection_async<T>(src, src_offset, dst, dst_offset);
    starpu_task_wait_for_all();
}

template<typename T>
void copy_intersection(const Tile<T> &src, const Tile<T> &dst)
{
    copy_intersection_async<T>(src, src.offset, dst, dst.offset);
    starpu_task_wait_for_all();
}

} // namespace nntile

