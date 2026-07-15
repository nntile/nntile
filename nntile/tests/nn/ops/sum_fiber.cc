/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/nn_graph/sum_fiber.cc
 * Test NNGraph sum_fiber autograd operation.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#ifdef NNTILE_HAVE_TORCH
#include "pytorch_helper.hh"
#include "pytorch_tile_helpers.hh"
#endif

#include "context_fixture.hh"
#include "nntile/graph.hh"

using namespace nntile;
using namespace nntile;
namespace gt = nntile::tensor;

namespace
{

constexpr Index batch_ndim_none = 0;
constexpr int redux_none = 0;
constexpr Index dim_2 = 2;
constexpr Index dim_4 = 4;

} // anonymous namespace

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph sum_fiber structure",
    "[graph][nn_graph]")
{
    const auto [x_shape, axis, alpha] =
        GENERATE(std::tuple{std::vector<Index>{dim_4, dim_2}, Index(1), Scalar(1.0)},
            std::tuple{std::vector<Index>{dim_4, dim_2}, Index(0), Scalar(2.0)});

    std::vector<Index> y_shape = {x_shape[axis]};

    NNGraph g("sum_fiber_structure");
    auto *x = g.tensor(x_shape, DataType::FP32)->set_name("x");
    auto *y =
        sum_fiber(x, axis, batch_ndim_none, redux_none, alpha)->set_name("y");

    REQUIRE(y != nullptr);
    REQUIRE(y->has_producer());
    REQUIRE(y->shape() == y_shape);
    REQUIRE(g.num_ops() >= 1);
    bool has_sum_fiber = false;
    for (const auto &op : g.tensor_graph().ops())
    {
        if (op->op_name() == "SUM_FIBER")
        {
            has_sum_fiber = true;
            break;
        }
    }
    REQUIRE(has_sum_fiber);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph sum_fiber backward",
    "[graph][nn_graph]")
{
    const auto [x_shape, axis, alpha, grad_fill_val] =
        GENERATE(std::tuple{std::vector<Index>{dim_4, dim_2},
                     Index(1),
                     Scalar(1.0),
                     Scalar(1.0)},
            std::tuple{std::vector<Index>{dim_4, dim_2},
                Index(0),
                Scalar(1.0),
                Scalar(-1.0)});

    std::vector<Index> y_shape = {x_shape[axis]};

    NNGraph g("sum_fiber_backward");
    auto *x = g.tensor(x_shape, DataType::FP32)->set_name("x");
    auto *y =
        sum_fiber(x, axis, batch_ndim_none, redux_none, alpha)->set_name("y");

    auto [y_grad, _] = g.get_or_create_grad(y, "y_grad");
    gt::fill(grad_fill_val, y_grad->data());
    y->backward();

    REQUIRE(x->has_grad());
    REQUIRE(x->grad()->shape() == x_shape);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph sum_fiber forward and backward",
    "[graph][nn_graph]")
{
    const auto [x_shape, axis, alpha, grad_fill_val] = GENERATE(
        std::tuple{
            std::vector<Index>{4, 2}, Index(1), Scalar(1.0), Scalar(1.0)},
        std::tuple{
            std::vector<Index>{4, 2}, Index(0), Scalar(1.0), Scalar(1.0)},
        std::tuple{
            std::vector<Index>{6, 3}, Index(0), Scalar(2.0), Scalar(1.0)},
        std::tuple{
            std::vector<Index>{6, 3}, Index(1), Scalar(1.0), Scalar(-1.0)},
        std::tuple{
            std::vector<Index>{4, 3, 2}, Index(2), Scalar(1.0), Scalar(1.0)});

    std::vector<Index> y_shape = {x_shape[axis]};

    NNGraph g("sum_fiber");
    auto *x = g.tensor(x_shape, DataType::FP32, true)->set_name("x");
    auto *y =
        sum_fiber(x, axis, batch_ndim_none, redux_none, alpha)->set_name("y");

    REQUIRE(y != nullptr);
    REQUIRE(y->has_producer());
    REQUIRE(y->shape() == y_shape);

    auto [y_grad, _] = g.get_or_create_grad(y, "y_grad");
    gt::fill(grad_fill_val, y_grad->data());
    y->backward();

    REQUIRE(x->has_grad());
    REQUIRE(x->grad()->shape() == x_shape);
}

#ifdef NNTILE_HAVE_TORCH

using nntile::test::compare_float_vectors;
using nntile::test::nn_pytorch_tile_heterogeneous_1d_len6;
using nntile::test::nn_pytorch_tile_heterogeneous_1d_len7;
using nntile::test::nn_pytorch_tile_heterogeneous_rank2_6x7;
using nntile::test::pytorch_tolerance;

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph sum_fiber forward matches PyTorch",
    "[graph][nn_graph][pytorch]")
{
    const auto [x_shape, axis, alpha] =
        GENERATE(std::tuple{std::vector<Index>{6, 7}, Index(1), Scalar(1.0)},
            std::tuple{std::vector<Index>{6, 7}, Index(0), Scalar(2.0)},
            std::tuple{std::vector<Index>{6, 7}, Index(1), Scalar(0.5)},
            std::tuple{std::vector<Index>{6, 7}, Index(0), Scalar(1.0)});

    const Index x_nelems = x_shape[0] * x_shape[1];
    const Index y_nelems = x_shape[axis];

    std::vector<float> x_data(x_nelems);
    for (Index i = 0; i < x_nelems; ++i)
        x_data[i] = 0.1f * static_cast<float>(i + 1);

    NNGraph g("sum_fiber_pytorch");
    auto *x = g.tensor(x_shape, DataType::FP32, true)->set_name("x");
    auto *y =
        sum_fiber(x, axis, batch_ndim_none, redux_none, alpha)->set_name("y");

    nn_pytorch_tile_heterogeneous_rank2_6x7(x);
    if (axis == 0)
        nn_pytorch_tile_heterogeneous_1d_len6(y);
    else
        nn_pytorch_tile_heterogeneous_1d_len7(y);

    TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());
    Runtime runtime(tile_graph);
    runtime.compile();
    runtime.bind_data(x, x_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> nntile_out = runtime.get_output<float>(y);
    std::vector<::int64_t> x_shape_pt(x_shape.begin(), x_shape.end());

    auto x_pt = torch::from_blob(x_data.data(),
        x_shape_pt,
        torch::TensorOptions().dtype(torch::kFloat32))
                    .clone()
                    .set_requires_grad(false);
    const auto ndim = static_cast<::int64_t>(x_shape.size());
    const auto dim_pt = ndim - 1 - static_cast<::int64_t>(axis);
    auto y_pt = (alpha * x_pt.sum(dim_pt, /*keepdim=*/false)).contiguous();
    std::vector<float> pytorch_out(
        y_pt.data_ptr<float>(), y_pt.data_ptr<float>() + y_nelems);

    REQUIRE(nntile_out.size() == pytorch_out.size());
    for (size_t i = 0; i < nntile_out.size(); ++i)
        REQUIRE(std::abs(nntile_out[i] - pytorch_out[i]) < pytorch_tolerance);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph sum_fiber backward matches PyTorch",
    "[graph][nn_graph][pytorch]")
{
    const auto [x_shape, axis, alpha, grad_fill_val] = GENERATE(
        std::tuple{
            std::vector<Index>{6, 7}, Index(1), Scalar(1.0), Scalar(1.0)},
        std::tuple{
            std::vector<Index>{6, 7}, Index(0), Scalar(1.0), Scalar(-1.0)},
        std::tuple{
            std::vector<Index>{6, 7}, Index(0), Scalar(2.0), Scalar(1.0)});

    const Index x_nelems = x_shape[0] * x_shape[1];
    const Index y_nelems = x_shape[axis];

    std::vector<float> x_data(x_nelems);
    for (Index i = 0; i < x_nelems; ++i)
        x_data[i] = 0.15f * static_cast<float>(i);

    NNGraph g("sum_fiber_bwd_pytorch");
    auto *x = g.tensor(x_shape, DataType::FP32, true)->set_name("x");
    auto *y =
        sum_fiber(x, axis, batch_ndim_none, redux_none, alpha)->set_name("y");

    nn_pytorch_tile_heterogeneous_rank2_6x7(x);
    if (axis == 0)
        nn_pytorch_tile_heterogeneous_1d_len6(y);
    else
        nn_pytorch_tile_heterogeneous_1d_len7(y);

    auto [y_grad, _] = g.get_or_create_grad(y, "y_grad");
    gt::fill(grad_fill_val, y_grad->data());
    y->backward();

    TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());
    Runtime runtime(tile_graph);
    runtime.compile();
    runtime.bind_data(x, x_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> nntile_grad_x = runtime.get_output<float>(x->grad());

    std::vector<::int64_t> x_shape_pt(x_shape.begin(), x_shape.end());

    auto x_pt = torch::from_blob(x_data.data(),
        x_shape_pt,
        torch::TensorOptions().dtype(torch::kFloat32))
                    .clone()
                    .set_requires_grad(true);
    const auto ndim = static_cast<::int64_t>(x_shape.size());
    const auto dim_pt = ndim - 1 - static_cast<::int64_t>(axis);
    auto y_pt = alpha * x_pt.sum(dim_pt, /*keepdim=*/false);
    auto grad_output = torch::full({static_cast<::int64_t>(y_nelems)},
        static_cast<float>(grad_fill_val),
        torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false));
    y_pt.backward(grad_output);

    compare_float_vectors(nntile_grad_x, x_pt.grad());
}

#endif // NNTILE_HAVE_TORCH
