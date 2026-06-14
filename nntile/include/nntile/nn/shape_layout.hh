/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/nn/shape_layout.hh
 * Virtual C-order shape labels for NNGraph (physical storage stays Fortran).
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>

#include <vector>

namespace nntile::nn
{

//! Reverse dimension labels: C [dimn..dim0] <-> Fortran [dim0..dimn].
inline std::vector<Index> reverse_shape(const std::vector<Index> &shape)
{
    return std::vector<Index>(shape.rbegin(), shape.rend());
}

//! C-order axis (0 = outermost) -> Fortran axis (0 = innermost).
inline Index c_axis_to_fortran(Index c_axis, Index ndim)
{
    return ndim - 1 - c_axis;
}

//! Fortran axis -> C-order axis.
inline Index fortran_axis_to_c(Index f_axis, Index ndim)
{
    return ndim - 1 - f_axis;
}

//! User C-order shape -> physical Fortran shape for tensor::*.
inline std::vector<Index> c_shape_to_fortran(const std::vector<Index> &c_shape)
{
    return reverse_shape(c_shape);
}

//! Physical Fortran shape -> virtual C-order shape for NNGraph::TensorNode.
inline std::vector<Index> fortran_shape_to_c(
    const std::vector<Index> &f_shape)
{
    return reverse_shape(f_shape);
}

} // namespace nntile::nn
