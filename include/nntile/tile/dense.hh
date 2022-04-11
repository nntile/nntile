#pragma once

// Get tile definition
#include <nntile/tile/tile.hh>

namespace nntile
{

template<typename T>
void from_dense(const Tile<T> &dst,
        const T *src,
        const std::vector<size_t> &stride)
{
}

} // namespace nntile

