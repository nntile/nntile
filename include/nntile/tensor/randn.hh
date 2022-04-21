#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile
{

template<typename T>
void randn_async(const Tensor<T> &dst, const std::vector<Index> &offset,
        const std::vector<Index> &shape, const std::vector<Index> &stride,
        unsigned long long seed, T mean=0, T stddev=1);

extern template
void randn_async(const Tensor<float> &dst, const std::vector<Index> &offset,
        const std::vector<Index> &shape, const std::vector<Index> &stride,
        unsigned long long seed, float mean=0, float stddev=1);

extern template
void randn_async(const Tensor<double> &dst, const std::vector<Index> &offset,
        const std::vector<Index> &shape, const std::vector<Index> &stride,
        unsigned long long seed, double mean=0, double stddev=1);

template<typename T>
void randn_async(const Tensor<T> &dst, unsigned long long seed, T mean=0,
        T stddev=1)
{
    randn_async<T>(dst, std::vector<Index>(dst.ndim), dst.shape, dst.stride,
            seed, mean, stddev);
}

template<typename T>
void randn(const Tensor<T> &dst, const std::vector<Index> &offset,
        const std::vector<Index> &shape, const std::vector<Index> &stride,
        unsigned long long seed, T mean=0, T stddev=1)
{
    randn_async<T>(dst, offset, shape, stride, seed, mean, stddev);
    starpu_task_wait_for_all();
}

template<typename T>
void randn(const Tensor<T> &dst, unsigned long long seed, T mean=0, T stddev=1)
{
    randn_async<T>(dst, std::vector<Index>(dst.ndim), dst.shape, dst.stride,
            seed, mean, stddev);
    starpu_task_wait_for_all();
}

} // namespace nntile

