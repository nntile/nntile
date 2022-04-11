#pragma once

#include <nntile/tile.hh>

namespace nntile
{

template<typename T>
void randn_async(const Tile<T> &A, const std::vector<size_t> &offset,
        const std::vector<size_t> &stride, unsigned long long &seed,
        T mean, T stddev);

extern template
void randn_async(const Tile<float> &A, const std::vector<size_t> &offset,
        const std::vector<size_t> &stride, unsigned long long &seed,
        float mean, float stddev);

extern template
void randn_async(const Tile<double> &A, const std::vector<size_t> &offset,
        const std::vector<size_t> &stride, unsigned long long &seed,
        double mean, double stddev);

template<typename T>
void randn(const Tile<T> &A, const std::vector<size_t> &offset,
        const std::vector<size_t> &stride,
        unsigned long long &seed, T mean, T stddev)
{
    randn_async<T>(A, offset, stride, seed, mean, stddev);
    starpu_task_wait_for_all();
}

} // namespace nntile

