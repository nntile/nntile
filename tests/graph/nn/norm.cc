/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/nn/norm.cc
 * Test NNGraph norm autograd operation.
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

constexpr Index dim_2 = 2;
constexpr Index dim_3 = 3;

} // anonymous namespace

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph norm structure", "[graph][nn_graph]")
{
    const auto alpha = GENERATE(Scalar(1.0), Scalar(2.0), Scalar(0.5));

    NNGraph g("norm_structure");
    auto* x = g.tensor({dim_2, dim_3}, "x", DataType::FP32);
    auto* y = norm(x, "y", alpha);

    REQUIRE(y != nullptr);
    REQUIRE(y->has_producer());
    REQUIRE(y->shape() == std::vector<Index>{});
    REQUIRE(g.num_ops() == 1);
    bool has_norm = false;
    for(const auto& op : g.tensor_graph().ops())
    {
        if(op->op_name() == "NORM")
        {
            has_norm = true;
            break;
        }
    }
    REQUIRE(has_norm);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph norm forward", "[graph][nn_graph]")
{
    const auto alpha = GENERATE(Scalar(1.0), Scalar(2.0), Scalar(0.5));

    NNGraph g("norm_forward");
    auto* x = g.tensor({dim_2, dim_3}, "x", DataType::FP32, false);
    auto* y = norm(x, "y", alpha);

    x->mark_input(true);
    y->mark_output(true);

    std::vector<float> x_data(dim_2 * dim_3);
    for(Index i = 0; i < dim_2 * dim_3; ++i)
        x_data[i] = static_cast<float>(i + 1);

    TensorGraph::Runtime runtime(g.tensor_graph());
    runtime.compile();
    runtime.bind_data("x", x_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> out = runtime.get_output<float>("y");
    REQUIRE(out.size() == 1);
    REQUIRE(out[0] > 0.0);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph norm backward throws", "[graph][nn_graph]")
{
    const auto [alpha, grad_fill_val] = GENERATE(
        std::tuple{Scalar(1.0), Scalar(1.0)},
        std::tuple{Scalar(2.0), Scalar(-1.0)},
        std::tuple{Scalar(0.5), Scalar(0.5)});

    NNGraph g("norm_backward_throws");
    auto* x = g.tensor({dim_2, dim_3}, "x", DataType::FP32, true);
    auto* y = norm(x, "y", alpha);

    auto [y_grad, _] = g.get_or_create_grad(y, "y_grad");
    gt::fill(grad_fill_val, y_grad->data());

    REQUIRE_THROWS_AS(y->backward(), std::runtime_error);
}
