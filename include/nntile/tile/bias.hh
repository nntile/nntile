#pragma once

#include <nntile/tile/tile.hh>

namespace nntile
{

template<typename T>
void bias_async(const Tile<T> &src, const Tile<T> &dst, Index batch_dim);

extern template
void bias_async(const Tile<float> &src, const Tile<float> &dst,
        Index batch_dim);

extern template
void bias_async(const Tile<double> &src, const Tile<double> &dst,
        Index batch_dim);

template<typename T>
void bias(const Tile<T> &src, const Tile<T> &dst, Index batch_dim)
{
    bias_async<T>(src, dst, batch_dim);
    starpu_task_wait_for_all();
}

} // namespace nntile

