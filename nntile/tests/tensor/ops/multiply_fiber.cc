/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/tensor_graph/multiply_fiber.cc
 * Test TensorGraph multiply_fiber operation.
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/ops/multiply_fiber.hh"

#include "context_fixture.hh"
#include "nntile/tensor.hh"
#include "nntile/tensor/axis_descriptor.hh"
#include "nntile/tile.hh"
#include "nntile/tensor/ops/multiply_fiber.hh"
#include "nntile/tensor.hh"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <numeric>

using namespace nntile;
using namespace nntile;
namespace gt = nntile::tensor;

namespace
{

constexpr Index axis_0 = 0;
constexpr Index axis_1 = 1;
constexpr Index axis_2 = 2;
constexpr Scalar alpha_one = 1.0;
constexpr Scalar alpha_half = 0.5;
constexpr Scalar alpha_two = 2.0;
constexpr float tolerance = 1e-5f;
constexpr int distr_rank_single = 0;

constexpr Index dim_2 = 2;
constexpr Index dim_3 = 3;
constexpr Index dim_4 = 4;
constexpr Index dim_5 = 5;

} // anonymous namespace

//! Fiber shape for multiply_fiber: {tensor_shape[axis]} (1D fiber)
static std::vector<Index> fiber_shape(
    const std::vector<Index> &tensor_shape, Index axis)
{
    return {tensor_shape[axis]};
}

TEST_CASE("TensorGraph multiply_fiber structure", "[graph][tensor]")
{
    TensorGraph graph("test");

    auto *fiber = graph.data({dim_4})->set_name("fiber");
    auto *tensor = graph.data({dim_4, dim_2})->set_name("tensor");

    auto *out =
        gt::multiply_fiber(alpha_one, fiber, tensor, axis_0)->set_name("out");

    REQUIRE(graph.num_data() == 3);
    REQUIRE(graph.num_ops() == 1);
    REQUIRE(out->shape() == (std::vector<Index>{dim_4, dim_2}));

    const auto &ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "MULTIPLY_FIBER");
    REQUIRE(ops[0]->inputs().size() == 2);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == out);
}

TEST_CASE(
    "TensorGraph multiply_fiber rejects duplicate tensors", "[graph][tensor]")
{
    TensorGraph graph("test");
    auto *fiber = graph.data({dim_4})->set_name("fiber");
    auto *tensor = graph.data({dim_4, dim_2})->set_name("tensor");

    REQUIRE_THROWS_AS(
        gt::multiply_fiber(alpha_one, fiber, tensor, tensor, axis_0),
        std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph multiply_fiber tiled matches untiled",
    "[graph][tensor]")
{
    const auto [tensor_shape, axis, alpha] =
        GENERATE(std::tuple{std::vector<Index>{4, 2}, Index(1), 1.0},
            std::tuple{std::vector<Index>{4, 2}, Index(0), 1.0});

    using T = nntile::fp32_t;
    using Y = T::repr_t;
    std::vector<Index> fiber_sh = fiber_shape(tensor_shape, axis);
    const Index tensor_nelems = std::accumulate(tensor_shape.begin(),
        tensor_shape.end(),
        Index(1),
        std::multiplies<>());
    const Index fiber_nelems = std::accumulate(
        fiber_sh.begin(), fiber_sh.end(), Index(1), std::multiplies<>());

    std::vector<float> fiber_data(fiber_nelems);
    std::vector<float> tensor_data(tensor_nelems);
    for (Index i = 0; i < fiber_nelems; ++i)
        fiber_data[i] = static_cast<float>(Y(i + 1));
    for (Index i = 0; i < tensor_nelems; ++i)
        tensor_data[i] = static_cast<float>(Y(-i - 1));

    std::vector<float> untiled_result;
    {
        TensorGraph graph("multiply_fiber_untiled");
        auto *fiber_node =
            graph.data(fiber_sh, DataType::FP32)->set_name("fiber");
        auto *tensor_node =
            graph.data(tensor_shape, DataType::FP32)->set_name("tensor");
        fiber_node->mark_input(true);
        tensor_node->mark_input(true);
        auto *out_node =
            gt::multiply_fiber(alpha, fiber_node, tensor_node, axis)
                ->set_name("out");
        out_node->mark_output(true);
        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);

        Runtime runtime(tile_graph);
        runtime.compile();
        runtime.bind_data(fiber_node, fiber_data);
        runtime.bind_data(tensor_node, tensor_data);
        runtime.execute();
        runtime.wait();
        untiled_result = runtime.get_output<float>(out_node);
    }

    std::vector<float> tiled_result;
    {
        TensorGraph graph("multiply_fiber_tiled");
        auto *fiber_node =
            graph.data(fiber_sh, DataType::FP32)->set_name("fiber");
        auto *tensor_node =
            graph.data(tensor_shape, DataType::FP32)->set_name("tensor");
        fiber_node->mark_input(true);
        tensor_node->mark_input(true);
        auto *out_node =
            gt::multiply_fiber(alpha, fiber_node, tensor_node, axis)
                ->set_name("out");
        out_node->mark_output(true);
        for (auto *ag : graph.axis_groups())
        {
            ag->set_tiling((ag->extent + 1) / 2);
        }
        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);

        Runtime runtime(tile_graph);
        runtime.compile();
        runtime.bind_data(fiber_node, fiber_data);
        runtime.bind_data(tensor_node, tensor_data);
        runtime.execute();
        runtime.wait();
        tiled_result = runtime.get_output<float>(out_node);
    }

    constexpr float tol = 1e-5f;
    REQUIRE(tiled_result.size() == untiled_result.size());
    for (size_t i = 0; i < tiled_result.size(); ++i)
    {
        REQUIRE(std::abs(tiled_result[i] - untiled_result[i]) < tol);
    }
}