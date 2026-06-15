/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/nn_graph/pytorch_helper.hh
 * Shared helpers for NNGraph PyTorch comparison tests: relative Frobenius error
 * (vector–vector and vector–tensor), element-wise relative checks.
 *
 * @version 1.1.0
 * */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include <catch2/catch_test_macros.hpp>

namespace nntile::test
{

//! Floor for relative scales (tiny activations / gradients).
constexpr float relative_tolerance_floor = 1e-7f;

//! \f$\|a-b\|_F / \max(\|a\|_F,\|b\|_F,\epsilon)\f$ (symmetric relative error).
//!
//! Using only \f$\|b\|_F\f$ blows up when the reference norm is tiny. ``b`` is
//! the nominal reference for documentation; the scale is symmetric in ``a`` and ``b``.
inline float relative_frobenius_error(
    const std::vector<float>& a,
    const std::vector<float>& b,
    float epsilon = relative_tolerance_floor)
{
    double sq_diff = 0.0;
    double sq_a = 0.0;
    double sq_b = 0.0;
    for(size_t i = 0; i < a.size(); ++i)
    {
        const double ai = static_cast<double>(a[i]);
        const double bi = static_cast<double>(b[i]);
        const double d = ai - bi;
        sq_diff += d * d;
        sq_a += ai * ai;
        sq_b += bi * bi;
    }
    const double na = std::sqrt(sq_a);
    const double nb = std::sqrt(sq_b);
    const double diff = std::sqrt(sq_diff);
    const double scale = std::max(
        na, std::max(nb, static_cast<double>(epsilon)));
    return static_cast<float>(diff / scale);
}

//! \f$\|a-b\|_F / \max(\|a\|_F,\|b\|_F,\epsilon)\f$ must be below ``tol``.
inline void require_relative_frobenius_error(const std::vector<float>& a,
    const std::vector<float>& b,
    float tol,
    float epsilon = relative_tolerance_floor)
{
    REQUIRE(relative_frobenius_error(a, b, epsilon) < tol);
}

} // namespace nntile::test

#ifdef NNTILE_HAVE_TORCH

#include <torch/torch.h>

#include <nntile/common.hh>

namespace nntile::test
{

using nntile::Index;

//! Default relative tolerance for float comparison with PyTorch (element and Frobenius).
constexpr float pytorch_tolerance = 1e-5f;

//! Extra relative slack for element-wise checks (autograd / reordering vs ``atol``).
constexpr float pytorch_element_rtol = 5e-4f;

inline float relative_frobenius_error(
    const std::vector<float>& a,
    const torch::Tensor& b,
    float epsilon = relative_tolerance_floor)
{
    REQUIRE(b.defined());
    REQUIRE(b.dtype() == torch::kFloat32);
    REQUIRE(static_cast<size_t>(b.numel()) == a.size());

    auto const b_flat = b.contiguous().view(-1);
    auto const b_acc = b_flat.accessor<float, 1>();
    double sq_diff = 0.0;
    double sq_a = 0.0;
    double sq_b = 0.0;
    for(size_t i = 0; i < a.size(); ++i)
    {
        const double ai = static_cast<double>(a[i]);
        const double bi = static_cast<double>(b_acc[static_cast<long>(i)]);
        const double d = ai - bi;
        sq_diff += d * d;
        sq_a += ai * ai;
        sq_b += bi * bi;
    }
    const double na = std::sqrt(sq_a);
    const double nb = std::sqrt(sq_b);
    const double diff = std::sqrt(sq_diff);
    const double scale = std::max(
        na, std::max(nb, static_cast<double>(epsilon)));
    return static_cast<float>(diff / scale);
}

//! \f$\|a-b\|_F / \max(\|a\|_F,\|b\|_F,\epsilon)\f$ vs flattened PyTorch tensor ``ref``.
inline void require_relative_frobenius_error(const std::vector<float>& got,
    const torch::Tensor& ref,
    float tol = pytorch_tolerance,
    float epsilon = relative_tolerance_floor)
{
    REQUIRE(relative_frobenius_error(got, ref, epsilon) < tol);
}

//! Per-element check: \f$|a_i-b_i| \le \texttt{tol} + \texttt{rtol}\cdot\max(|a_i|,|b_i|,\epsilon)\f$.
//!
//! Pure relative error with a tiny floor blows up when both sides are near zero (only
//! numerical noise differs). The absolute term ``tol`` matches the legacy absolute test
//! for that regime; ``rtol`` covers normal float32 Jacobian noise.
inline void compare_float_vectors(const std::vector<float>& a,
                                  const torch::Tensor& b,
                                  float tol = pytorch_tolerance,
                                  float epsilon = relative_tolerance_floor,
                                  float rtol = pytorch_element_rtol)
{
    REQUIRE(b.defined());
    REQUIRE(b.dtype() == torch::kFloat32);
    REQUIRE(static_cast<size_t>(b.numel()) == a.size());

    auto const b_flat = b.contiguous().view(-1);
    auto const b_acc = b_flat.accessor<float, 1>();
    for(size_t i = 0; i < a.size(); ++i)
    {
        const double ai = static_cast<double>(a[i]);
        const double bi = static_cast<double>(b_acc[static_cast<long>(i)]);
        const double abs_diff = std::fabs(ai - bi);
        const double scale = std::max(
            std::fabs(ai), std::max(std::fabs(bi), static_cast<double>(epsilon)));
        const double bound =
            static_cast<double>(tol) + static_cast<double>(rtol) * scale;
        REQUIRE(abs_diff <= bound);
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

} // namespace nntile::test

#endif // NNTILE_HAVE_TORCH
