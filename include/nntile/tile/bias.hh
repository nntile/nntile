#pragma once

#include <nntile/tile/tile.hh>

namespace nntile
{

template<typename T>
void bias_async(const Tile<T> &A, const Tile<T> &bias, int batch_dim);

extern template
void bias_async(const Tile<float> &A, const Tile<float> &bias,
        int batch_dim);

extern template
void bias_async(const Tile<double> &A, const Tile<double> &bias,
        int batch_dim);

template<typename T>
void bias(const Tile<T> &A, const Tile<T> &bias, int batch_dim)
{
    bias_async<T>(A, bias, batch_dim);
    starpu_task_wait_for_all();
}

} // namespace nntile

