/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file include/nntile/core/swap_two_axes_decompose.hh
 * N-D to 5D decomposition for swap_two_axes.
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>

#include <array>
#include <stdexcept>
#include <vector>

namespace nntile::core
{

struct SwapTwoAxesDecomposition
{
    std::array<Index, 5> sizes_5d{};
    std::vector<Index> output_shape;
};

inline SwapTwoAxesDecomposition decompose_swap_axes(
    const std::vector<Index> &shape,
    Index dim0,
    Index dim1)
{
    const Index n = static_cast<Index>(shape.size());
    if (n < 2)
    {
        throw std::invalid_argument(
            "decompose_swap_axes: shape rank must be >= 2");
    }
    if (dim0 < 0 || dim1 < 0 || dim0 >= n || dim1 >= n)
    {
        throw std::invalid_argument(
            "decompose_swap_axes: axis out of range");
    }
    if (dim0 == dim1)
    {
        throw std::invalid_argument(
            "decompose_swap_axes: axes must differ");
    }
    if (dim0 > dim1)
    {
        std::swap(dim0, dim1);
    }

    Index d0 = 1;
    for (Index i = 0; i < dim0; ++i)
    {
        d0 *= shape[static_cast<size_t>(i)];
    }
    const Index d1 = shape[static_cast<size_t>(dim0)];
    Index d2 = 1;
    for (Index i = dim0 + 1; i < dim1; ++i)
    {
        d2 *= shape[static_cast<size_t>(i)];
    }
    const Index d3 = shape[static_cast<size_t>(dim1)];
    Index d4 = 1;
    for (Index i = dim1 + 1; i < n; ++i)
    {
        d4 *= shape[static_cast<size_t>(i)];
    }

    std::vector<Index> output_shape = shape;
    std::swap(
        output_shape[static_cast<size_t>(dim0)],
        output_shape[static_cast<size_t>(dim1)]);

    SwapTwoAxesDecomposition out;
    out.sizes_5d = {d0, d1, d2, d3, d4};
    out.output_shape = std::move(output_shape);
    return out;
}

} // namespace nntile::core
