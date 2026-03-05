/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/nn/norm_slice.cc
 * Test NNGraph norm_slice autograd operation.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "context_fixture.hh"
#include "nntile/graph.hh"

using namespace nntile;
using namespace nntile::graph;
namespace gt = nntile::graph::tensor;

namespace
{

constexpr int redux_none = 0;
constexpr Index dim_2 = 2;
constexpr Index dim_4 = 4;

} // anonymous namespace

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph norm_slice structure", "[graph][nn_graph]")
{
    const auto [alpha, axis] = GENERATE(
        std::tuple{Scalar(1.0), Index(0)},
        std::tuple{Scalar(1.0), Index(1)},
        std::tuple{Scalar(2.0), Index(0)},
        std::tuple{Scalar(0.5), Index(1)});

    std::vector<Index> x_shape = (axis == 0) ?
        std::vector<Index>{dim_4, dim_2} : std::vector<Index>{dim_2, dim_4};
    std::vector<Index> out_shape;
    for(Index i = 0; i < static_cast<Index>(x_shape.size()); ++i)
        if(i != axis)
            out_shape.push_back(x_shape[i]);

    NNGraph g("norm_slice_structure");
    auto* x = g.tensor(x_shape, "x", DataType::FP32);
    auto* y = norm_slice(x, "y", axis, redux_none, alpha);

    REQUIRE(y != nullptr);
    REQUIRE(y->has_producer());
    REQUIRE(y->shape() == out_shape);
    REQUIRE(g.num_ops() >= 1);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph norm_slice forward", "[graph][nn_graph]")
{
    const auto [alpha, axis] = GENERATE(
        std::tuple{Scalar(1.0), Index(1)},
        std::tuple{Scalar(2.0), Index(0)},
        std::tuple{Scalar(0.5), Index(1)});

    NNGraph g("norm_slice_forward");
    std::vector<Index> x_shape = (axis == 0) ?
        std::vector<Index>{dim_4, dim_2} : std::vector<Index>{dim_2, dim_4};
    auto* x = g.tensor(x_shape, "x", DataType::FP32, false);
    auto* y = norm_slice(x, "y", axis, redux_none, alpha);

    x->mark_input(true);
    y->mark_output(true);

    Index x_nelems = 1;
    for(Index d : x_shape)
        x_nelems *= d;
    std::vector<float> x_data(static_cast<size_t>(x_nelems));
    for(Index i = 0; i < x_nelems; ++i)
        x_data[i] = static_cast<float>(i + 1);

    TensorGraph::Runtime runtime(g.tensor_graph());
    runtime.compile();
    runtime.bind_data("x", x_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> out = runtime.get_output<float>("y");
    REQUIRE(out.size() == static_cast<size_t>(dim_2));
    for(float v : out)
        REQUIRE(v > 0.0);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph norm_slice backward throws", "[graph][nn_graph]")
{
    const auto [alpha, axis, grad_fill_val] = GENERATE(
        std::tuple{Scalar(1.0), Index(1), Scalar(1.0)},
        std::tuple{Scalar(2.0), Index(0), Scalar(-1.0)},
        std::tuple{Scalar(0.5), Index(1), Scalar(0.5)});

    std::vector<Index> x_shape = (axis == 0) ?
        std::vector<Index>{dim_4, dim_2} : std::vector<Index>{dim_2, dim_4};
    NNGraph g("norm_slice_backward_throws");
    auto* x = g.tensor(x_shape, "x", DataType::FP32, true);
    auto* y = norm_slice(x, "y", axis, redux_none, alpha);

    auto [y_grad, _] = g.get_or_create_grad(y, "y_grad");
    gt::fill(grad_fill_val, y_grad->data());

    REQUIRE_THROWS_AS(y->backward(), std::runtime_error);
}
