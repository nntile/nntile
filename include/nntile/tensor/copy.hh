#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile
{

template<typename T>
void copy_intersection_async(const Tensor<T> &src,
        const std::vector<Index> &src_offset,
        const Tensor<T> &dst, const std::vector<Index> &dst_offset);

extern template
void copy_intersection_async(const Tensor<float> &src,
        const std::vector<Index> &src_offset, const Tensor<float> &dst,
        const std::vector<Index> &dst_offset);

extern template
void copy_intersection_async(const Tensor<double> &src,
        const std::vector<Index> &src_offset, const Tensor<double> &dst,
        const std::vector<Index> &dst_offset);

template<typename T>
void copy_intersection_async(const Tensor<T> &src, const Tensor<T> &dst)
{
    copy_intersection_async<T>(src, std::vector<Index>(src.ndim), dst,
            std::vector<Index>(dst.ndim));
}

template<typename T>
void copy_intersection(const Tensor<T> &src,
        const std::vector<Index> &src_offset, const Tensor<T> &dst,
        const std::vector<Index> &dst_offset)
{
    copy_intersection_async<T>(src, src_offset, dst, dst_offset);
    starpu_task_wait_for_all();
}

template<typename T>
void copy_intersection(const Tensor<T> &src, const Tensor<T> &dst)
{
    copy_intersection_async<T>(src, std::vector<Index>(src.ndim), dst,
            std::vector<Index>(dst.ndim));
    starpu_task_wait_for_all();
}

} // namespace nntile

