#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile
{

template<typename T>
void randn_async(const TileTraits &src, const Tensor<T> &dst,
        const std::vector<size_t> &dst_coord, unsigned long long seed,
        T mean=0, T stddev=1);

extern template
void randn_async(const TileTraits &src, const Tensor<float> &dst,
        const std::vector<size_t> &dst_coord, unsigned long long seed,
        float mean=0, float stddev=1);

extern template
void randn_async(const TileTraits &src, const Tensor<double> &dst,
        const std::vector<size_t> &dst_coord, unsigned long long seed,
        double mean=0, double stddev=1);

template<typename T>
void randn_async(const Tensor<T> &A, unsigned long long seed, T mean=0,
        T stddev=1)
{
    randn_async<T>(A, A, std::vector<size_t>(A.ndim, 0), seed, mean, stddev);
}

template<typename T>
void randn(const TileTraits &src, const Tensor<T> &dst,
        const std::vector<size_t> &dst_coord, unsigned long long seed,
        T mean=0, T stddev=1)
{
    randn_async<T>(src, dst, dst_coord, seed, mean, stddev);
    starpu_task_wait_for_all();
}

template<typename T>
void randn(const Tensor<T> &A, unsigned long long seed, T mean=0, T stddev=1)
{
    randn_async<T>(A, A, std::vector<size_t>(A.ndim, 0), seed, mean, stddev);
    starpu_task_wait_for_all();
}

} // namespace nntile

