/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/tensor_graph/gemm.cc
 * Test TensorGraph gemm operation.
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/ops/gemm.hh"

#include "context_fixture.hh"
#include "gemm_core_reference.hh"
#include "gemm_test_shapes.hh"
#include "nntile/constants.hh"
#include "nntile/tensor.hh"
#include "nntile/tensor/axis_descriptor.hh"
#include "nntile/tile.hh"
#include "nntile/tensor/ops/clear.hh"
#include "nntile/tensor/ops/gemm.hh"
#include "nntile/tensor.hh"
#include "test_frobenius.hh"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <cmath>
#include <numeric>

using namespace nntile;
namespace gt = nntile::tensor;

namespace
{

constexpr Scalar alpha_one = 1.0;
constexpr bool trans_a = false;
constexpr bool trans_b = false;
constexpr Index ndim = 1;
constexpr Index batch_ndim = 0;

using nntile::test::gemm_relative_tolerance;
using nntile::test::require_relative_frobenius_error;

std::vector<float> run_gemm_graph(const std::vector<Index> &a_shape,
    const std::vector<Index> &b_shape,
    const std::vector<float> &a_data,
    const std::vector<float> &b_data,
    Scalar alpha,
    bool trans_a_flag,
    bool trans_b_flag,
    Index ndim_flag,
    Index batch_ndim_flag,
    bool tiled)
{
    TensorGraph graph(tiled ? "gemm_tiled" : "gemm_untiled");
    auto *a_node = graph.data(a_shape, DataType::FP32)->set_name("a");
    auto *b_node = graph.data(b_shape, DataType::FP32)->set_name("b");
    a_node->mark_input(true);
    b_node->mark_input(true);

    auto *c_node = gt::gemm(a_node,
        b_node,
        alpha,
        trans_a_flag,
        trans_b_flag,
        ndim_flag,
        batch_ndim_flag);
    c_node->mark_output(true);

    if (tiled)
    {
        for (auto *ag : graph.axis_groups())
        {
            ag->set_tiling((ag->extent + 1) / 2);
        }
    }

    TileGraph tile_graph = TileGraph::from_tensor_graph(graph);
    Runtime runtime(tile_graph);
    runtime.compile();
    runtime.bind_data(a_node, a_data);
    runtime.bind_data(b_node, b_data);
    runtime.execute();
    runtime.wait();
    return runtime.get_output<float>(c_node);
}

void require_tiled_matches_untiled(const std::vector<float> &untiled,
    const std::vector<float> &tiled)
{
    require_relative_frobenius_error(tiled, untiled, gemm_relative_tolerance);
}

} // namespace

TEST_CASE("TensorGraph gemm structure", "[graph][tensor]")
{
    TensorGraph graph("test");

    auto *a = graph.data({5, 4})->set_name("a");
    auto *b = graph.data({4, 6})->set_name("b");
    auto *c = gt::gemm(a, b, alpha_one, trans_a, trans_b, ndim, batch_ndim);

    REQUIRE(graph.num_data() == 3);
    REQUIRE(graph.num_ops() == 1);
    REQUIRE(c->shape().size() == 2);
    REQUIRE(c->shape()[0] == 5);
    REQUIRE(c->shape()[1] == 6);

    const auto &ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "GEMM");
    REQUIRE(ops[0]->inputs().size() == 3);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == c);
}

TEST_CASE("TensorGraph gemm rejects null", "[graph][tensor]")
{
    TensorGraph graph("test");
    auto *a = graph.data({5, 4})->set_name("a");
    auto *b = graph.data({5, 6})->set_name("b");

    REQUIRE_THROWS_AS(
        gt::gemm(nullptr, b, alpha_one, trans_a, trans_b, ndim, batch_ndim),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        gt::gemm(a, nullptr, alpha_one, trans_a, trans_b, ndim, batch_ndim),
        std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph gemm tiled matches untiled",
    "[graph][tensor]")
{
    const auto [trans_a_flag, trans_b_flag, ndim_flag, batch_ndim_flag, alpha] =
        GENERATE(
            std::tuple{false, false, Index(1), Index(0), Scalar(1.0)},
            std::tuple{false, true, Index(1), Index(0), Scalar(1.0)},
            std::tuple{true, false, Index(1), Index(0), Scalar(1.0)},
            std::tuple{true, true, Index(1), Index(0), Scalar(1.0)},
            std::tuple{false, false, Index(2), Index(0), Scalar(0.5)},
            std::tuple{false, true, Index(2), Index(0), Scalar(0.5)},
            std::tuple{true, false, Index(2), Index(0), Scalar(0.5)},
            std::tuple{true, true, Index(2), Index(0), Scalar(0.5)},
            std::tuple{false, false, Index(1), Index(1), Scalar(1.0)},
            std::tuple{false, true, Index(1), Index(1), Scalar(1.0)},
            std::tuple{true, false, Index(1), Index(1), Scalar(1.0)},
            std::tuple{true, true, Index(1), Index(1), Scalar(1.0)},
            std::tuple{false, false, Index(1), Index(2), Scalar(1.0)},
            std::tuple{false, true, Index(1), Index(2), Scalar(1.0)},
            std::tuple{true, false, Index(1), Index(2), Scalar(1.0)},
            std::tuple{true, true, Index(1), Index(2), Scalar(1.0)});

    const auto [a_shape, b_shape] = nntile::test::gemm_test_shapes(
        trans_a_flag, trans_b_flag, ndim_flag, batch_ndim_flag);
    const Index a_nelems = std::accumulate(a_shape.begin(),
        a_shape.end(),
        Index{1},
        std::multiplies<Index>{});
    const Index b_nelems = std::accumulate(b_shape.begin(),
        b_shape.end(),
        Index{1},
        std::multiplies<Index>{});

    using Y = nntile::fp32_t::repr_t;
    std::vector<float> a_data(static_cast<size_t>(a_nelems));
    std::vector<float> b_data(static_cast<size_t>(b_nelems));
    for (Index i = 0; i < a_nelems; ++i)
    {
        a_data[static_cast<size_t>(i)] =
            static_cast<float>(Y(i % 10)) * 0.1f;
    }
    for (Index i = 0; i < b_nelems; ++i)
    {
        b_data[static_cast<size_t>(i)] =
            static_cast<float>(Y(i % 7)) * 0.15f;
    }

    const std::vector<float> untiled = run_gemm_graph(a_shape,
        b_shape,
        a_data,
        b_data,
        alpha,
        trans_a_flag,
        trans_b_flag,
        ndim_flag,
        batch_ndim_flag,
        false);
    const std::vector<float> tiled = run_gemm_graph(a_shape,
        b_shape,
        a_data,
        b_data,
        alpha,
        trans_a_flag,
        trans_b_flag,
        ndim_flag,
        batch_ndim_flag,
        true);

    require_tiled_matches_untiled(untiled, tiled);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph gemm matches core",
    "[graph][tensor]")
{
    const auto [trans_a_flag, trans_b_flag, ndim_flag, batch_ndim_flag, alpha] =
        GENERATE(
            std::tuple{false, false, Index(1), Index(0), Scalar(1.0)},
            std::tuple{false, true, Index(1), Index(0), Scalar(1.0)},
            std::tuple{true, false, Index(1), Index(0), Scalar(1.0)},
            std::tuple{true, true, Index(1), Index(0), Scalar(1.0)},
            std::tuple{false, false, Index(2), Index(0), Scalar(0.5)},
            std::tuple{false, true, Index(2), Index(0), Scalar(0.5)},
            std::tuple{true, false, Index(2), Index(0), Scalar(0.5)},
            std::tuple{true, true, Index(2), Index(0), Scalar(0.5)},
            std::tuple{false, false, Index(1), Index(1), Scalar(1.0)},
            std::tuple{false, true, Index(1), Index(1), Scalar(1.0)},
            std::tuple{true, false, Index(1), Index(1), Scalar(1.0)},
            std::tuple{true, true, Index(1), Index(1), Scalar(1.0)},
            std::tuple{false, false, Index(1), Index(2), Scalar(1.0)},
            std::tuple{false, true, Index(1), Index(2), Scalar(1.0)},
            std::tuple{true, false, Index(1), Index(2), Scalar(1.0)},
            std::tuple{true, true, Index(1), Index(2), Scalar(1.0)});

    const auto [a_shape, b_shape] = nntile::test::gemm_test_shapes(
        trans_a_flag, trans_b_flag, ndim_flag, batch_ndim_flag);
    const Index a_nelems = std::accumulate(a_shape.begin(),
        a_shape.end(),
        Index{1},
        std::multiplies<Index>{});
    const Index b_nelems = std::accumulate(b_shape.begin(),
        b_shape.end(),
        Index{1},
        std::multiplies<Index>{});

    using Y = nntile::fp32_t::repr_t;
    std::vector<float> a_data(static_cast<size_t>(a_nelems));
    std::vector<float> b_data(static_cast<size_t>(b_nelems));
    for (Index i = 0; i < a_nelems; ++i)
    {
        a_data[static_cast<size_t>(i)] =
            static_cast<float>(Y(i % 10)) * 0.1f;
    }
    for (Index i = 0; i < b_nelems; ++i)
    {
        b_data[static_cast<size_t>(i)] =
            static_cast<float>(Y(i % 7)) * 0.15f;
    }

    const std::vector<float> core_out = nntile::test::core_gemm_reference_fp32(
        a_shape,
        b_shape,
        a_data,
        b_data,
        alpha,
        trans_a_flag,
        trans_b_flag,
        ndim_flag,
        batch_ndim_flag);
    const std::vector<float> tensor_out = run_gemm_graph(a_shape,
        b_shape,
        a_data,
        b_data,
        alpha,
        trans_a_flag,
        trans_b_flag,
        ndim_flag,
        batch_ndim_flag,
        false);

    require_relative_frobenius_error(tensor_out, core_out, gemm_relative_tolerance);
}
