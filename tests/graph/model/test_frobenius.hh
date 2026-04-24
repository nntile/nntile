/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file tests/graph/model/test_frobenius.hh
 * Relative Frobenius-norm checks for flattened float tensors in graph tests.
 *
 * @version 1.1.0
 * */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

//! \f$ \|a-b\|_F / \max(\|a\|_F,\|b\|_F,\epsilon)\f$ (symmetric relative error).
//!
//! Using only \f$\|b\|_F\f$ blows up when the reference norm is tiny (common
//! for gradients or sparse activations). ``b`` is still the nominal reference
//! for documentation; the scale is symmetric in ``a`` and ``b``.
inline float relative_frobenius_error(
    const std::vector<float>& a,
    const std::vector<float>& b,
    float epsilon = 1e-7f)
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
