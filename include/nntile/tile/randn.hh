#pragma once

#include <nntile/tile/tile.hh>

namespace nntile
{

template<typename T>
void randn_async(const Tile<T> &dst, unsigned long long seed, T mean=0,
        T stddev=1);

extern template
void randn_async(const Tile<float> &dst, unsigned long long seed,
        float mean=0, float stddev=1);

extern template
void randn_async(const Tile<double> &dst, unsigned long long seed,
        double mean=0, double stddev=1);

template<typename T>
void randn(const Tile<T> &A, unsigned long long seed, T mean=0, T stddev=1)
{
    randn_async<T>(A, seed, mean, stddev);
    starpu_task_wait_for_all();
}

} // namespace nntile

