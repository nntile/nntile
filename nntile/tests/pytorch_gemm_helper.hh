/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/pytorch_gemm_helper.hh
 * PyTorch matmul / batched matmul reference for NNTile C-order GEMM.
 *
 * @version 1.1.0
 * */

#pragma once

#ifdef NNTILE_HAVE_TORCH

#   include <cstdint>
#   include <vector>

#   include <torch/torch.h>

#   include <nntile/common.hh>
#   include <nntile/tensor/ops/gemm.hh>

#   include "gemm_test_shapes.hh"

namespace nntile::test
{

using nntile::Index;
using nntile::Scalar;

namespace detail
{

inline ::int64_t product_extent(const std::vector<::int64_t> &shape,
    Index begin,
    Index end)
{
    ::int64_t prod = 1;
    for (Index i = begin; i < end; ++i)
    {
        prod *= shape[static_cast<size_t>(i)];
    }
    return prod;
}

inline std::vector<::int64_t> slice_sizes(const std::vector<::int64_t> &shape,
    Index begin,
    Index end)
{
    std::vector<::int64_t> out;
    out.reserve(static_cast<size_t>(end - begin));
    for (Index i = begin; i < end; ++i)
    {
        out.push_back(shape[static_cast<size_t>(i)]);
    }
    return out;
}

inline std::vector<::int64_t> concat_sizes(
    const std::vector<::int64_t> &a,
    const std::vector<::int64_t> &b)
{
    std::vector<::int64_t> out = a;
    out.insert(out.end(), b.begin(), b.end());
    return out;
}

struct GemmAxisRanges
{
    Index a_k_begin = 0;
    Index a_k_end = 0;
    Index a_m_begin = 0;
    Index a_m_end = 0;
    Index b_k_begin = 0;
    Index b_k_end = 0;
    Index b_n_begin = 0;
    Index b_n_end = 0;
};

inline GemmAxisRanges gemm_axis_ranges(Index a_ndim,
    Index b_ndim,
    bool trans_a,
    bool trans_b,
    Index ndim,
    Index batch_ndim)
{
    GemmAxisRanges r;
    if (trans_a)
    {
        r.a_m_begin = batch_ndim;
        r.a_m_end = a_ndim - ndim;
        r.a_k_begin = a_ndim - ndim;
        r.a_k_end = a_ndim;
    }
    else
    {
        r.a_k_begin = batch_ndim;
        r.a_k_end = batch_ndim + ndim;
        r.a_m_begin = batch_ndim + ndim;
        r.a_m_end = a_ndim;
    }
    if (trans_b)
    {
        r.b_n_begin = batch_ndim;
        r.b_n_end = b_ndim - ndim;
        r.b_k_begin = b_ndim - ndim;
        r.b_k_end = b_ndim;
    }
    else
    {
        r.b_k_begin = batch_ndim;
        r.b_k_end = batch_ndim + ndim;
        r.b_n_begin = batch_ndim + ndim;
        r.b_n_end = b_ndim;
    }
    return r;
}

inline torch::Tensor permute_to_gemm_layout(const torch::Tensor &t,
    const GemmAxisRanges &ranges,
    bool is_a,
    Index batch_ndim)
{
    const auto sizes = t.sizes().vec();
    const Index ndim = static_cast<Index>(sizes.size());
    std::vector<::int64_t> perm;
    perm.reserve(static_cast<size_t>(ndim));
    for (Index i = 0; i < batch_ndim; ++i)
    {
        perm.push_back(i);
    }
    if (is_a)
    {
        if (ranges.a_k_begin < ranges.a_m_begin)
        {
            for (Index i = ranges.a_k_begin; i < ranges.a_k_end; ++i)
                perm.push_back(i);
            for (Index i = ranges.a_m_begin; i < ranges.a_m_end; ++i)
                perm.push_back(i);
        }
        else
        {
            for (Index i = ranges.a_m_begin; i < ranges.a_m_end; ++i)
                perm.push_back(i);
            for (Index i = ranges.a_k_begin; i < ranges.a_k_end; ++i)
                perm.push_back(i);
        }
    }
    else
    {
        if (ranges.b_k_begin < ranges.b_n_begin)
        {
            for (Index i = ranges.b_k_begin; i < ranges.b_k_end; ++i)
                perm.push_back(i);
            for (Index i = ranges.b_n_begin; i < ranges.b_n_end; ++i)
                perm.push_back(i);
        }
        else
        {
            for (Index i = ranges.b_n_begin; i < ranges.b_n_end; ++i)
                perm.push_back(i);
            for (Index i = ranges.b_k_begin; i < ranges.b_k_end; ++i)
                perm.push_back(i);
        }
    }
    return t.permute(perm);
}

} // namespace detail

//! PyTorch reference for NNTile C-order GEMM (``matmul`` / ``bmm``).
inline torch::Tensor pytorch_gemm_reference(const torch::Tensor &a,
    const torch::Tensor &b,
    bool trans_a,
    bool trans_b,
    Index ndim,
    Index batch_ndim,
    Scalar alpha = 1.0)
{
    const auto a_sizes = a.sizes().vec();
    const auto b_sizes = b.sizes().vec();
    const Index a_ndim = static_cast<Index>(a_sizes.size());
    const Index b_ndim = static_cast<Index>(b_sizes.size());

    const detail::GemmAxisRanges ranges = detail::gemm_axis_ranges(
        a_ndim, b_ndim, trans_a, trans_b, ndim, batch_ndim);

    const ::int64_t batch_prod = detail::product_extent(a_sizes, 0, batch_ndim);
    const ::int64_t k_prod = detail::product_extent(
        a_sizes, ranges.a_k_begin, ranges.a_k_end);
    const ::int64_t m_prod = detail::product_extent(
        a_sizes, ranges.a_m_begin, ranges.a_m_end);
    const ::int64_t n_prod = detail::product_extent(
        b_sizes, ranges.b_n_begin, ranges.b_n_end);

    torch::Tensor a_perm = detail::permute_to_gemm_layout(
        a, ranges, true, batch_ndim);
    torch::Tensor b_perm = detail::permute_to_gemm_layout(
        b, ranges, false, batch_ndim);

    torch::Tensor a_flat;
    torch::Tensor b_flat;
    if (trans_a)
    {
        a_flat = a_perm.reshape({batch_prod, m_prod, k_prod});
    }
    else
    {
        a_flat = a_perm.reshape({batch_prod, k_prod, m_prod});
    }
    if (trans_b)
    {
        b_flat = b_perm.reshape({batch_prod, n_prod, k_prod});
    }
    else
    {
        b_flat = b_perm.reshape({batch_prod, k_prod, n_prod});
    }

    torch::Tensor c_flat;
    if (batch_ndim == 0)
    {
        torch::Tensor a2 = a_flat.squeeze(0);
        torch::Tensor b2 = b_flat.squeeze(0);
        if (!trans_a && !trans_b)
        {
            c_flat = torch::matmul(b2.transpose(0, 1), a2);
        }
        else if (trans_a && !trans_b)
        {
            c_flat = torch::matmul(a2, b2).transpose(0, 1);
        }
        else if (!trans_a && trans_b)
        {
            c_flat = torch::matmul(b2, a2);
        }
        else
        {
            c_flat = torch::matmul(b2, a2.transpose(0, 1));
        }
    }
    else
    {
        if (!trans_a && !trans_b)
        {
            c_flat = torch::bmm(b_flat.transpose(1, 2), a_flat);
        }
        else if (trans_a && !trans_b)
        {
            c_flat = torch::bmm(a_flat, b_flat).transpose(1, 2);
        }
        else if (!trans_a && trans_b)
        {
            c_flat = torch::bmm(b_flat, a_flat);
        }
        else
        {
            c_flat = torch::bmm(b_flat, a_flat.transpose(1, 2));
        }
    }

  {
    const std::vector<Index> out_shape = nntile::tensor::gemm_output_shape(
        std::vector<Index>(a_sizes.begin(), a_sizes.end()),
        std::vector<Index>(b_sizes.begin(), b_sizes.end()),
        trans_a,
        trans_b,
        ndim,
        batch_ndim);
    std::vector<::int64_t> out_sizes(out_shape.begin(), out_shape.end());
    return (static_cast<double>(alpha) * c_flat).reshape(out_sizes).contiguous();
  }
}

} // namespace nntile::test

#endif // NNTILE_HAVE_TORCH
