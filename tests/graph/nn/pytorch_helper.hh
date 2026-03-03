/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/nn/pytorch_helper.hh
 * Shared helpers for NNGraph PyTorch comparison tests.
 *
 * @version 1.1.0
 * */

#pragma once

#ifdef NNTILE_HAVE_TORCH

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <vector>

#include <torch/torch.h>

#include "nntile/base_types.hh"

namespace nntile::test
{

//! Default tolerance for float comparison with PyTorch
constexpr float pytorch_tolerance = 1e-5f;

//! Compare NNTile output (vector) with PyTorch tensor element-wise
inline void compare_float_vectors(const std::vector<float>& a,
                                  const torch::Tensor& b,
                                  float tol = pytorch_tolerance)
{
    REQUIRE(b.defined());
    REQUIRE(b.dtype() == torch::kFloat32);
    REQUIRE(static_cast<size_t>(b.numel()) == a.size());

    auto b_flat = b.contiguous().view(-1);
    auto b_acc = b_flat.accessor<float, 1>();
    for(size_t i = 0; i < a.size(); ++i)
    {
        REQUIRE(std::abs(a[i] - b_acc[static_cast<long>(i)]) < tol);
    }
}

//! Broadcast 1D fiber along axis to match tensor shape (for PyTorch)
inline torch::Tensor broadcast_fiber(const torch::Tensor& fiber,
                                    torch::IntArrayRef tensor_shape,
                                    Index axis)
{
    std::vector<::int64_t> dims(tensor_shape.size(), 1);
    dims[axis] = -1;
    return fiber.view(dims).expand(tensor_shape);
}

//! Convert column-major (Fortran) to row-major for comparison with PyTorch
template<typename T>
inline std::vector<T> colmajor_to_rowmajor(const std::vector<T>& data,
                                          const std::vector<Index>& shape)
{
    std::vector<T> result(data.size());
    const Index ndim = static_cast<Index>(shape.size());
    std::vector<Index> row_strides(ndim);
    row_strides[ndim - 1] = 1;
    for(Index i = ndim - 2; i >= 0; --i)
        row_strides[i] = row_strides[i + 1] * shape[i + 1];
    for(Index col_idx = 0; col_idx < static_cast<Index>(data.size()); ++col_idx)
    {
        Index idx = col_idx;
        Index row_idx = 0;
        for(Index d = 0; d < ndim; ++d)
        {
            Index coord = idx % shape[d];
            row_idx += coord * row_strides[d];
            idx /= shape[d];
        }
        result[row_idx] = data[col_idx];
    }
    return result;
}

//! Permute data from row-major layout. perm[i] = source dim for output dim i.
//! E.g. perm {2,0,1} on shape [M,N,B] yields [B,M,N] layout.
template<typename T>
inline std::vector<T> permute_rowmajor(const std::vector<T>& data,
                                      const std::vector<Index>& shape,
                                      const std::vector<Index>& perm)
{
    const Index ndim = static_cast<Index>(shape.size());
    std::vector<Index> out_shape(ndim);
    for(Index i = 0; i < ndim; ++i)
        out_shape[i] = shape[perm[i]];
    std::vector<Index> in_strides(ndim);
    in_strides[ndim - 1] = 1;
    for(Index i = ndim - 2; i >= 0; --i)
        in_strides[i] = in_strides[i + 1] * shape[i + 1];
    std::vector<Index> out_strides(ndim);
    out_strides[ndim - 1] = 1;
    for(Index i = ndim - 2; i >= 0; --i)
        out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
    std::vector<T> result(data.size());
    std::vector<Index> coord(ndim);
    for(Index out_idx = 0; out_idx < static_cast<Index>(data.size()); ++out_idx)
    {
        Index idx = out_idx;
        for(Index d = ndim - 1; d >= 0; --d)
        {
            coord[d] = idx % out_shape[d];
            idx /= out_shape[d];
        }
        Index in_idx = 0;
        for(Index d = 0; d < ndim; ++d)
            in_idx += coord[d] * in_strides[perm[d]];
        result[out_idx] = data[in_idx];
    }
    return result;
}

} // namespace nntile::test

#endif // NNTILE_HAVE_TORCH
