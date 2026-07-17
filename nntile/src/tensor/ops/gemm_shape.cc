/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file src/tensor/ops/gemm_shape.cc
 * Shape helper for GEMM (shared by classic and torch-native builds).
 *
 * @version 1.1.0
 */

#include "nntile/tensor/ops/gemm.hh"

#include <vector>

namespace nntile::tensor
{

std::vector<Index> gemm_output_shape(
    const std::vector<Index> &a_shape,
    const std::vector<Index> &b_shape,
    bool trans_a,
    bool trans_b,
    Index ndim,
    Index batch_ndim)
{
    Index a_ndim = static_cast<Index>(a_shape.size());
    Index b_ndim = static_cast<Index>(b_shape.size());

    const Index a_batch_end = batch_ndim;
    const Index a_k_begin = trans_a ? batch_ndim : (a_ndim - ndim);
    const Index a_m_begin = trans_a ? (batch_ndim + ndim) : batch_ndim;
    const Index a_m_end = trans_a ? a_ndim : (a_ndim - ndim);
    const Index b_n_begin = trans_b ? batch_ndim : (batch_ndim + ndim);
    const Index b_n_end = trans_b ? (b_ndim - ndim) : b_ndim;
    (void)a_k_begin;

    std::vector<Index> output_shape;
    output_shape.reserve(static_cast<size_t>(
        batch_ndim + (a_m_end - a_m_begin) + (b_n_end - b_n_begin)));
    output_shape.insert(
        output_shape.end(),
        a_shape.begin(),
        a_shape.begin() + a_batch_end);
    output_shape.insert(
        output_shape.end(),
        a_shape.begin() + a_m_begin,
        a_shape.begin() + a_m_end);
    output_shape.insert(
        output_shape.end(),
        b_shape.begin() + b_n_begin,
        b_shape.begin() + b_n_end);
    return output_shape;
}

} // namespace nntile::tensor
