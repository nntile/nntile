/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/tensor_graph/norm_fiber.cc
 * Test TensorGraph norm_fiber operation.
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/ops/norm_fiber.hh"

#include "context_fixture.hh"
#include "nntile/tensor.hh"
#include "nntile/tensor/axis_descriptor.hh"
#include "nntile/tile.hh"
#include "nntile/tensor/ops/norm_fiber.hh"
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
constexpr Index batch_ndim_none = 0;
constexpr int redux_none = 0;
constexpr Scalar alpha_one = 1.0;
constexpr Scalar alpha_two = 2.0;
constexpr Scalar beta_zero = 0.0;
constexpr Scalar beta_half = 0.5;
constexpr float y_init_overwrite = 0.0f;
constexpr float y_init_accumulate = 1.0f;
constexpr float tolerance = 1e-5f;
constexpr int distr_rank_single = 0;
constexpr Index x_fill_offset = 1;

constexpr Index dim_2 = 2;
constexpr Index dim_3 = 3;
constexpr Index dim_4 = 4;
constexpr Index dim_5 = 5;
constexpr Index dim_6 = 6;

} // anonymous namespace

//! Output shape for norm_fiber: {x_shape[axis]} for batch_ndim=0
static std::vector<Index> norm_fiber_output_shape(
    const std::vector<Index> &x_shape, Index axis, Index batch_ndim)
{
    std::vector<Index> out_shape;
    out_shape.reserve(batch_ndim + 1);
    for (Index i = 0; i < batch_ndim; ++i)
    {
        out_shape.push_back(x_shape[i]);
    }
    out_shape.push_back(x_shape[axis]);
    return out_shape;
}

TEST_CASE("TensorGraph norm_fiber structure", "[graph][tensor]")
{
    TensorGraph graph("test");

    nntile::TensorRef x = graph.data({dim_4, dim_5});
    x->set_name("x");
    nntile::TensorRef y = graph.data({dim_4});
    y->set_name("y");

    nntile::TensorRef out = nntile::TensorRef::adopt(gt::norm_fiber(
        alpha_one, x, beta_zero, y, axis_0, batch_ndim_none, redux_none)
                    );
    out->set_name("out");

    REQUIRE(graph.num_data() == 3);
    REQUIRE(graph.num_ops() == 1);
    REQUIRE(out->shape().size() == 1);
    REQUIRE(out->shape()[0] == dim_4);

    const auto &ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "NORM_FIBER");
    REQUIRE(ops[0]->inputs().size() == 2);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == out);
}

TEST_CASE(
    "TensorGraph norm_fiber rejects duplicate tensors", "[graph][tensor]")
{
    TensorGraph graph("test");
    nntile::TensorRef x = graph.data({dim_4, dim_5});
    x->set_name("x");
    nntile::TensorRef y = graph.data({dim_4});
    y->set_name("y");

    REQUIRE_THROWS_AS(gt::norm_fiber(alpha_one,
                          x,
                          beta_zero,
                          y,
                          y,
                          axis_0,
                          batch_ndim_none,
                          redux_none),
        std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph norm_fiber tiled matches untiled",
    "[graph][tensor]")
{
    const auto [x_shape, axis, batch_ndim, redux, alpha, beta] =
        GENERATE(std::tuple{std::vector<Index>{dim_4, dim_5},
                     axis_0,
                     batch_ndim_none,
                     redux_none,
                     alpha_one,
                     beta_zero},
            std::tuple{std::vector<Index>{dim_2, dim_3, dim_4},
                axis_1,
                batch_ndim_none,
                redux_none,
                alpha_one,
                beta_zero});

    using T = nntile::fp32_t;
    using Y = typename T::repr_t;
    const Index x_nelems = std::accumulate(
        x_shape.begin(), x_shape.end(), Index(1), std::multiplies<>());

    std::vector<Index> y_shape =
        norm_fiber_output_shape(x_shape, axis, batch_ndim);
    const Index y_nelems = std::accumulate(
        y_shape.begin(), y_shape.end(), Index(1), std::multiplies<>());

    std::vector<float> x_data(x_nelems);
    for (Index i = 0; i < x_nelems; ++i)
    {
        x_data[i] = static_cast<float>(Y(i + x_fill_offset));
    }
    std::vector<float> y_data(y_nelems);
    for (Index i = 0; i < y_nelems; ++i)
    {
        y_data[i] = (beta != beta_zero) ? y_init_accumulate : y_init_overwrite;
    }

    // --- Untiled run ---
    std::vector<float> untiled_result;
    {
        TensorGraph graph("norm_fiber_untiled");
        nntile::TensorRef x_node = graph.data(x_shape, DataType::FP32);
    x_node->set_name("x");
        nntile::TensorRef y_node = graph.data(y_shape, DataType::FP32);
    y_node->set_name("y");

        nntile::TensorRef out_node = nntile::TensorRef::adopt(gt::norm_fiber(
            alpha, x_node, beta, y_node, axis, batch_ndim, redux)
                             );
    out_node->set_name("out");

        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);

        Runtime runtime(tile_graph);
        runtime.compile();

        runtime.bind_data(x_node, x_data);
        runtime.bind_data(y_node, y_data);
        runtime.execute();
        runtime.wait();

        untiled_result = runtime.get_output<float>(out_node);
    }

    // --- Tiled run ---
    std::vector<float> tiled_result;
    {
        TensorGraph graph("norm_fiber_tiled");
        nntile::TensorRef x_node = graph.data(x_shape, DataType::FP32);
    x_node->set_name("x");
        nntile::TensorRef y_node = graph.data(y_shape, DataType::FP32);
    y_node->set_name("y");

        nntile::TensorRef out_node = nntile::TensorRef::adopt(gt::norm_fiber(
            alpha, x_node, beta, y_node, axis, batch_ndim, redux)
                             );
    out_node->set_name("out");
        for (auto *ag : graph.axis_groups())
        {
            ag->set_tiling((ag->extent + 1) / 2);
        }

        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);

        Runtime runtime(tile_graph);
        runtime.compile();

        runtime.bind_data(x_node, x_data);
        runtime.bind_data(y_node, y_data);
        runtime.execute();
        runtime.wait();

        tiled_result = runtime.get_output<float>(out_node);
    }

    // --- Compare ---
    constexpr float tol = 1e-5f;
    REQUIRE(tiled_result.size() == untiled_result.size());
    for (size_t i = 0; i < tiled_result.size(); ++i)
    {
        REQUIRE(std::abs(tiled_result[i] - untiled_result[i]) < tol);
    }
}
