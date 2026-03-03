/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/nn/add_fiber.cc
 * Test NNGraph add_fiber autograd operation.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "context_fixture.hh"
#include "nntile/graph.hh"

using namespace nntile;
using namespace nntile::graph;

namespace
{

constexpr Index batch_ndim_none = 0;
constexpr Index dim_2 = 2;
constexpr Index dim_4 = 4;

} // anonymous namespace

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph add_fiber structure", "[graph][nn_graph]")
{
    const auto [alpha, beta, axis] = GENERATE(
        std::tuple{Scalar(1.0), Scalar(1.0), Index(1)},
        std::tuple{Scalar(0.5), Scalar(2.0), Index(0)});

    NNGraph g("add_fiber_structure");
    auto* fiber = g.tensor({dim_4}, "fiber", DataType::FP32);
    auto* tensor = g.tensor({dim_2, dim_4}, "tensor", DataType::FP32);
    auto* out = add_fiber(alpha, fiber, beta, tensor, "out",
                         axis, batch_ndim_none);

    REQUIRE(out != nullptr);
    REQUIRE(out->has_producer());
    REQUIRE(out->shape() == (std::vector<Index>{dim_2, dim_4}));
    REQUIRE(g.num_ops() == 1);
    REQUIRE(g.tensor_graph().ops()[0]->op_name() == "ADD_FIBER");
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph add_fiber backward", "[graph][nn_graph]")
{
    const auto [alpha, beta, axis, grad_fill_val] = GENERATE(
        std::tuple{Scalar(1.0), Scalar(1.0), Index(1), Scalar(1.0)},
        std::tuple{Scalar(2.0), Scalar(0.0), Index(0), Scalar(-1.0)});

    std::vector<Index> tensor_shape = (axis == 0) ?
        std::vector<Index>{dim_4, dim_2} : std::vector<Index>{dim_2, dim_4};
    std::vector<Index> fiber_shape = {tensor_shape[axis]};

    NNGraph g("add_fiber_backward");
    auto* fiber = g.tensor(fiber_shape, "fiber", DataType::FP32);
    auto* tensor = g.tensor(tensor_shape, "tensor", DataType::FP32);
    auto* out = add_fiber(alpha, fiber, beta, tensor, "out",
                         axis, batch_ndim_none);

    auto* out_grad = g.get_or_create_grad(out, "out_grad");
    fill(grad_fill_val, out_grad->data());
    out->backward();

    REQUIRE(fiber->has_grad());
    REQUIRE(tensor->has_grad());
    REQUIRE(fiber->grad()->shape() == fiber_shape);
    REQUIRE(tensor->grad()->shape() == tensor_shape);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph add_fiber forward and backward", "[graph][nn_graph]")
{
    const auto [tensor_shape, fiber_len, axis, alpha, beta, grad_fill_val] =
        GENERATE(
            std::tuple{std::vector<Index>{2, 4}, Index(4), Index(1), Scalar(1.0),
                       Scalar(1.0), Scalar(1.0)},
            std::tuple{std::vector<Index>{2, 4}, Index(2), Index(0), Scalar(1.0),
                       Scalar(1.0), Scalar(1.0)},
            std::tuple{std::vector<Index>{3, 5}, Index(5), Index(1), Scalar(0.5),
                       Scalar(2.0), Scalar(1.0)},
            std::tuple{std::vector<Index>{3, 5}, Index(3), Index(0), Scalar(2.0),
                       Scalar(0.0), Scalar(-1.0)});

    std::vector<Index> fiber_shape = {fiber_len};

    NNGraph g("add_fiber");
    auto* fiber = g.tensor(fiber_shape, "fiber", DataType::FP32, true);
    auto* tensor = g.tensor(tensor_shape, "tensor", DataType::FP32, true);
    auto* out = add_fiber(alpha, fiber, beta, tensor, "out",
                         axis, batch_ndim_none);

    REQUIRE(out != nullptr);
    REQUIRE(out->has_producer());
    REQUIRE(out->shape() == tensor_shape);

    auto* out_grad = g.get_or_create_grad(out, "out_grad");
    fill(grad_fill_val, out_grad->data());
    out->backward();

    REQUIRE(fiber->has_grad());
    REQUIRE(tensor->has_grad());
    REQUIRE(fiber->grad()->shape() == fiber_shape);
    REQUIRE(tensor->grad()->shape() == tensor_shape);
}
