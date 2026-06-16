/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/gemm_core_reference.hh
 * Reference GEMM via nntile::core for tile/tensor parity tests.
 *
 * @version 1.1.0
 * */

#pragma once

#include <vector>

#include <starpu.h>

#include <nntile/constants.hh>
#include <nntile/core/gemm.hh>
#include <nntile/core/tile.hh>
#include <nntile/tensor/ops/gemm.hh>

namespace nntile::test
{

inline std::vector<float> core_gemm_reference_fp32(
    const std::vector<Index> &a_shape,
    const std::vector<Index> &b_shape,
    const std::vector<float> &a_data,
    const std::vector<float> &b_data,
    Scalar alpha,
    bool trans_a,
    bool trans_b,
    Index ndim,
    Index batch_ndim)
{
    using T = fp32_t;
    using Y = typename T::repr_t;

    const std::vector<Index> c_shape = nntile::tensor::gemm_output_shape(
        a_shape, b_shape, trans_a, trans_b, ndim, batch_ndim);

    nntile::core::Tile<T> ta(a_shape);
    nntile::core::Tile<T> tb(b_shape);
    nntile::core::Tile<T> tc(c_shape);
    {
        auto la = ta.acquire(STARPU_W);
        auto lb = tb.acquire(STARPU_W);
        auto lc = tc.acquire(STARPU_W);
        for (Index i = 0; i < ta.nelems; ++i)
        {
            la[i] = Y(a_data[static_cast<size_t>(i)]);
        }
        for (Index i = 0; i < tb.nelems; ++i)
        {
            lb[i] = Y(b_data[static_cast<size_t>(i)]);
        }
        for (Index i = 0; i < tc.nelems; ++i)
        {
            lc[i] = Y(0);
        }
        la.release();
        lb.release();
        lc.release();
    }

    const TransOp op_a(trans_a ? TransOp::Trans : TransOp::NoTrans);
    const TransOp op_b(trans_b ? TransOp::Trans : TransOp::NoTrans);
    nntile::core::gemm<T>(
        -1, alpha, op_a, ta, op_b, tb, Scalar(0), tc, ndim, batch_ndim, 0);
    starpu_task_wait_for_all();

    std::vector<float> out(static_cast<size_t>(tc.nelems));
    {
        auto lc = tc.acquire(STARPU_R);
        for (Index i = 0; i < tc.nelems; ++i)
        {
            out[static_cast<size_t>(i)] = static_cast<float>(lc[i]);
        }
        lc.release();
    }
    return out;
}

} // namespace nntile::test
