#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile
{

template<typename T>
void copy_async(const Tensor<T> &src, const std::vector<size_t> &src_coord,
        const Tensor<T> &dst, const std::vector<size_t> &dst_coord);

extern template
void copy_async(const Tensor<float> &src,
        const std::vector<size_t> &src_coord, const Tensor<float> &dst,
        const std::vector<size_t> &dst_coord);

extern template
void copy_async(const Tensor<double> &src,
        const std::vector<size_t> &src_coord, const Tensor<double> &dst,
        const std::vector<size_t> &dst_coord);

template<typename T>
void copy(const Tensor<T> &src, const std::vector<size_t> &src_coord,
        const Tensor<T> &dst, const std::vector<size_t> &dst_coord)
{
    copy_async<T>(src, src_coord, dst, dst_coord);
    starpu_task_wait_for_all();
}

} // namespace nntile

