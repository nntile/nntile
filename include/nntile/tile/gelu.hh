#pragma once

#include <nntile/tile/tile.hh>

namespace nntile
{

template<typename T>
void gelu_async(const Tile<T> &A);

extern template
void gelu_async(const Tile<float> &A);

extern template
void gelu_async(const Tile<double> &A);

template<typename T>
void gelu(const Tile<T> &A)
{
    gelu_async<T>(A);
    starpu_task_wait_for_all();
}

} // namespace nntile

