/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tile/mixed_tile_common.hh
 * Shared helpers for tensor vs TileGraph parity under mixed per-axis tile sizes.
 *
 * @version 1.1.0
 * */

#pragma once

#include <cmath>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <nntile/graph.hh>

namespace nntile::graph::tile_tests
{

constexpr float k_rel_eps = 1e-7f;

inline float max_rel_err(const std::vector<float>& a, const std::vector<float>& b)
{
    REQUIRE(a.size() == b.size());
    float m = 0.f;
    for(size_t i = 0; i < a.size(); ++i)
    {
        const float den = std::max(std::abs(a[i]), k_rel_eps);
        m = std::max(m, std::abs(a[i] - b[i]) / den);
    }
    return m;
}

inline float frob_rel_err(const std::vector<float>& a, const std::vector<float>& b)
{
    REQUIRE(a.size() == b.size());
    double num = 0.;
    double den = 0.;
    for(size_t i = 0; i < a.size(); ++i)
    {
        const double di = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        num += di * di;
        den += static_cast<double>(a[i]) * static_cast<double>(a[i]);
    }
    return static_cast<float>(
        std::sqrt(num) / std::max(std::sqrt(den), static_cast<double>(k_rel_eps)));
}

//! Two-axis mixed tile sizes (2D logical tensor).
inline void apply_mixed_tile_sizes_2d(TensorGraph::TensorNode* n)
{
    n->axis(0)->set_tiling(std::vector<Index>{2, 3, 5});
    n->axis(1)->set_tiling(std::vector<Index>{4, 8});
}

//! One-axis mixed tile sizes (1D logical tensor).
inline void apply_mixed_tile_sizes_1d(TensorGraph::TensorNode* n)
{
    n->axis(0)->set_tiling(std::vector<Index>{2, 3, 5});
}

} // namespace nntile::graph::tile_tests
