/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/shape_utils.hh
 * Shape / axis helpers for C-order tensor conventions.
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>

#include <vector>

namespace nntile
{

//! Reverse a shape vector (Fortran-labeled -> C-labeled and vice versa).
inline std::vector<Index> reverse_shape(const std::vector<Index> &shape)
{
    return std::vector<Index>(shape.rbegin(), shape.rend());
}

//! Remap axis index when reversing shape dimension order.
inline Index remap_axis_f_to_c(Index axis_f, Index ndim)
{
    return ndim - 1 - axis_f;
}

//! Remap axis index from C-order to Fortran-labeled order.
inline Index remap_axis_c_to_f(Index axis_c, Index ndim)
{
    return ndim - 1 - axis_c;
}

} // namespace nntile
