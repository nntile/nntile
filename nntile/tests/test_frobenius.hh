/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/test_frobenius.hh
 * Relative Frobenius helpers for graph / core parity tests.
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

//! Default relative Frobenius tolerance for graph parity checks.
constexpr float gemm_relative_tolerance = 1e-6f;

//! Default max per-element relative tolerance for fp32 parity checks.
constexpr float element_relative_tolerance = 1e-6f;

//! \f$\|a-b\|_F / \max(\|a\|_F,\|b\|_F,\epsilon)\f$ (symmetric relative error).
inline float relative_frobenius_error(
    const std::vector<float> &a,
    const std::vector<float> &b,
    float epsilon = relative_tolerance_floor)
{
    double sq_diff = 0.0;
    double sq_a = 0.0;
    double sq_b = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
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

//! \f$\max_i |a_i-b_i| / \max(|a_i|,|b_i|,\epsilon)\f$.
inline float max_element_relative_error(
    const std::vector<float> &a,
    const std::vector<float> &b,
    float epsilon = relative_tolerance_floor)
{
    REQUIRE(a.size() == b.size());
    float max_err = 0.f;
    for(size_t i = 0; i < a.size(); ++i)
    {
        const float diff = std::fabs(a[i] - b[i]);
        const float scale = std::max(
            std::fabs(a[i]),
            std::max(std::fabs(b[i]), epsilon));
        max_err = std::max(max_err, diff / scale);
    }
    return max_err;
}

inline void require_relative_element_error(const std::vector<float> &a,
    const std::vector<float> &b,
    float tol = element_relative_tolerance,
    float epsilon = relative_tolerance_floor)
{
    REQUIRE(max_element_relative_error(a, b, epsilon) < tol);
}

inline void require_relative_frobenius_error(const std::vector<float> &a,
    const std::vector<float> &b,
    float tol = gemm_relative_tolerance,
    float epsilon = relative_tolerance_floor)
{
    REQUIRE(relative_frobenius_error(a, b, epsilon) < tol);
}

} // namespace nntile::test
