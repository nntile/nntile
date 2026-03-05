/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/nn/multiply.cc
 * Test NNGraph multiply autograd operation.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#ifdef NNTILE_HAVE_TORCH
#   include "pytorch_helper.hh"
#endif

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
    "NNGraph multiply structure", "[graph][nn_graph]")
{
    const auto alpha = GENERATE(Scalar(1.0), Scalar(2.0), Scalar(0.5), Scalar(-1.0));

    NNGraph g("multiply_structure");
    auto* x = g.tensor({dim_2, dim_3}, "x", DataType::FP32);
    auto* y = g.tensor({dim_2, dim_3}, "y", DataType::FP32);
    auto* z = multiply(x, y, "z", alpha);

    REQUIRE(z != nullptr);
    REQUIRE(z->has_producer());
    REQUIRE(z->shape() == (std::vector<Index>{dim_2, dim_3}));
    REQUIRE(g.num_ops() == 1);
    REQUIRE(g.tensor_graph().ops()[0]->op_name() == "MULTIPLY");
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph multiply backward", "[graph][nn_graph]")
{
    const auto [alpha, grad_fill_val] = GENERATE(
        std::tuple{Scalar(1.0), Scalar(1.0)},
        std::tuple{Scalar(2.0), Scalar(-1.0)},
        std::tuple{Scalar(0.5), Scalar(0.5)},
        std::tuple{Scalar(-1.0), Scalar(2.0)});

    NNGraph g("multiply_backward");
    auto* x = g.tensor({dim_2, dim_3}, "x", DataType::FP32);
    auto* y = g.tensor({dim_2, dim_3}, "y", DataType::FP32);
    auto* z = multiply(x, y, "z", alpha);

    auto [z_grad, _] = g.get_or_create_grad(z, "z_grad");
    gt::fill(grad_fill_val, z_grad->data());
    z->backward();

    REQUIRE(x->has_grad());
    REQUIRE(y->has_grad());
}

#ifdef NNTILE_HAVE_TORCH

using nntile::test::compare_float_vectors;
using nntile::test::pytorch_tolerance;

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph multiply forward matches PyTorch", "[graph][nn_graph][pytorch]")
{
    const auto alpha = GENERATE(Scalar(1.0), Scalar(2.0), Scalar(0.5), Scalar(-1.0));

    constexpr Index dim0 = 4;
    constexpr Index dim1 = 6;
    constexpr Index nelems = dim0 * dim1;

    std::vector<float> x_data(nelems);
    std::vector<float> y_data(nelems);
    for(Index i = 0; i < nelems; ++i)
    {
        x_data[i] = 0.1f * static_cast<float>(i + 1);
        y_data[i] = 0.2f * static_cast<float>(-i - 1);
    }

    NNGraph g("multiply_pytorch");
    auto* x = g.tensor({dim0, dim1}, "x", DataType::FP32, true);
    auto* y = g.tensor({dim0, dim1}, "y", DataType::FP32, true);
    auto* z = multiply(x, y, "z", alpha);

    x->mark_input(true);
    y->mark_input(true);
    z->mark_output(true);

    TensorGraph::Runtime runtime(g.tensor_graph());
    runtime.compile();
    runtime.bind_data("x", x_data);
    runtime.bind_data("y", y_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> nntile_out = runtime.get_output<float>("z");

    auto x_pt = torch::from_blob(x_data.data(), {dim0, dim1},
                                 torch::TensorOptions().dtype(torch::kFloat32))
                    .clone().set_requires_grad(false);
    auto y_pt = torch::from_blob(y_data.data(), {dim0, dim1},
                                 torch::TensorOptions().dtype(torch::kFloat32))
                    .clone().set_requires_grad(false);

    auto z_pt = (alpha * x_pt * y_pt).contiguous();
    std::vector<float> pytorch_out(z_pt.data_ptr<float>(),
                                   z_pt.data_ptr<float>() + nelems);

    REQUIRE(nntile_out.size() == pytorch_out.size());
    for(size_t i = 0; i < nntile_out.size(); ++i)
        REQUIRE(std::abs(nntile_out[i] - pytorch_out[i]) < pytorch_tolerance);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph multiply backward matches PyTorch", "[graph][nn_graph][pytorch]")
{
    const auto [alpha, grad_fill_val] = GENERATE(
        std::tuple{Scalar(1.0), Scalar(1.0)},
        std::tuple{Scalar(2.0), Scalar(-1.0)},
        std::tuple{Scalar(0.5), Scalar(0.5)},
        std::tuple{Scalar(-1.0), Scalar(2.0)});

    constexpr Index dim0 = 3;
    constexpr Index dim1 = 5;
    constexpr Index nelems = dim0 * dim1;

    std::vector<float> x_data(nelems);
    std::vector<float> y_data(nelems);
    for(Index i = 0; i < nelems; ++i)
    {
        x_data[i] = 0.1f * static_cast<float>(i);
        y_data[i] = 0.15f * static_cast<float>(i + 10);
    }

    NNGraph g("multiply_bwd_pytorch");
    auto* x = g.tensor({dim0, dim1}, "x", DataType::FP32, true);
    auto* y = g.tensor({dim0, dim1}, "y", DataType::FP32, true);
    auto* z = multiply(x, y, "z", alpha);

    x->mark_input(true);
    y->mark_input(true);

    auto [z_grad, _] = g.get_or_create_grad(z, "z_grad");
    gt::fill(grad_fill_val, z_grad->data());
    z->backward();

    x->grad()->mark_output(true);
    y->grad()->mark_output(true);

    TensorGraph::Runtime runtime(g.tensor_graph());
    runtime.compile();
    runtime.bind_data("x", x_data);
    runtime.bind_data("y", y_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> nntile_grad_x =
        runtime.get_output<float>(x->grad()->name());
    std::vector<float> nntile_grad_y =
        runtime.get_output<float>(y->grad()->name());

    auto x_pt = torch::from_blob(x_data.data(), {dim0, dim1},
                                 torch::TensorOptions().dtype(torch::kFloat32))
                    .clone().set_requires_grad(true);
    auto y_pt = torch::from_blob(y_data.data(), {dim0, dim1},
                                 torch::TensorOptions().dtype(torch::kFloat32))
                    .clone().set_requires_grad(true);

    auto z_pt = alpha * x_pt * y_pt;
    auto grad_output = torch::full(
        {dim0, dim1}, static_cast<float>(grad_fill_val),
        torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false));
    z_pt.backward(grad_output);

    compare_float_vectors(nntile_grad_x, x_pt.grad());
    compare_float_vectors(nntile_grad_y, y_pt.grad());
}

#endif // NNTILE_HAVE_TORCH
