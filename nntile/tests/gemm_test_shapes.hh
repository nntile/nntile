/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/gemm_test_shapes.hh
 * Shared C-order GEMM operand shapes for graph tests.
 *
 * @version 1.1.0
 * */

#pragma once

#include <utility>
#include <vector>

#include <nntile/common.hh>

namespace nntile::test
{

using nntile::Index;

//! Build C-order operand shapes for ``ndim`` contraction dims.
inline std::pair<std::vector<Index>, std::vector<Index>> gemm_test_shapes(
    bool trans_a,
    bool trans_b,
    Index ndim,
    Index batch_ndim)
{
    const Index B1 = 2;
    const Index B2 = 3;
    const Index K1 = 4;
    const Index K2 = 3;
    const Index M1 = 2;
    const Index M2 = 3;
    const Index N1 = 5;
    const Index N2 = 2;

    std::vector<Index> batch_shape;
    if (batch_ndim == 1)
    {
        batch_shape = {B1};
    }
    else if (batch_ndim == 2)
    {
        batch_shape = {B1, B2};
    }

    std::vector<Index> k_shape;
    if (ndim == 1)
    {
        k_shape = {K1};
    }
    else
    {
        k_shape = {K1, K2};
    }

    std::vector<Index> m_shape;
    if (ndim == 1)
    {
        m_shape = {M1};
    }
    else
    {
        m_shape = {M1, M2};
    }

    std::vector<Index> n_shape;
    if (ndim == 1)
    {
        n_shape = {N1};
    }
    else
    {
        n_shape = {N1, N2};
    }

    std::vector<Index> a_shape = batch_shape;
    std::vector<Index> b_shape = batch_shape;
    if (trans_a)
    {
        a_shape.insert(a_shape.end(), k_shape.begin(), k_shape.end());
        a_shape.insert(a_shape.end(), m_shape.begin(), m_shape.end());
    }
    else
    {
        a_shape.insert(a_shape.end(), m_shape.begin(), m_shape.end());
        a_shape.insert(a_shape.end(), k_shape.begin(), k_shape.end());
    }
    if (trans_b)
    {
        b_shape.insert(b_shape.end(), n_shape.begin(), n_shape.end());
        b_shape.insert(b_shape.end(), k_shape.begin(), k_shape.end());
    }
    else
    {
        b_shape.insert(b_shape.end(), k_shape.begin(), k_shape.end());
        b_shape.insert(b_shape.end(), n_shape.begin(), n_shape.end());
    }
    return {a_shape, b_shape};
}

} // namespace nntile::test
