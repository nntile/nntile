/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/context_fixture.hh
 * Catch2 fixture providing Context for graph tests.
 *
 * @version 1.1.0
 * */

#pragma once

#include <cstddef>

#include "nntile/context.hh"

namespace nntile
{
namespace test
{

//! Catch2 fixture that constructs Context with named constexpr parameters
struct ContextFixture
{
    static constexpr int n_workers = 1;
    static constexpr int n_cuda = 0;
    static constexpr int ooc_enabled = 0;
    static constexpr char const* ooc_path = "/tmp/nntile_ooc";
    static constexpr std::size_t ooc_size = 16777216;
    static constexpr int logger = 0;

    Context context;

    ContextFixture()
        : context(n_workers, n_cuda, ooc_enabled, ooc_path, ooc_size, logger)
    {
    }
};

} // namespace test
} // namespace nntile
