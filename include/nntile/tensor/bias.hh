#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile
{

template<typename T>
void bias_async(const Tensor<T> &src, const Tensor<T> &dst, Index batch_dim);

extern template
void bias_async(const Tensor<float> &src, const Tensor<float> &dst,
        Index batch_dim);

extern template
void bias_async(const Tensor<double> &src, const Tensor<double> &dst,
        Index batch_dim);

template<typename T>
void bias(const Tensor<T> &src, const Tensor<T> &dst, Index batch_dim)
{
    bias_async<T>(src, dst, batch_dim);
    starpu_task_wait_for_all();
}

} // namespace nntile

