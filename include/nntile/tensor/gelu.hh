#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile
{

template<typename T>
void gelu_async(const Tensor<T> &A);

extern template
void gelu_async(const Tensor<float> &A);

extern template
void gelu_async(const Tensor<double> &A);

template<typename T>
void gelu(const Tensor<T> &A)
{
    gelu_async<T>(A);
    starpu_task_wait_for_all();
}

} // namespace nntile

