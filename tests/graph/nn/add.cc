/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/nn/add.cc
 * Test NNGraph add autograd operation.
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

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph add rejects shape mismatch", "[graph][nn_graph]")
{
    NNGraph g("add_shape_mismatch");
    auto* x = g.tensor({2, 3}, "x", DataType::FP32);
    auto* y = g.tensor({3, 2}, "y", DataType::FP32);  // different shape

    REQUIRE_THROWS_AS(add(1.0, x, 1.0, y, "z"), std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph add structure", "[graph][nn_graph]")
{
    const auto [alpha, beta] = GENERATE(
        std::tuple{Scalar(1.0), Scalar(1.0)},
        std::tuple{Scalar(2.0), Scalar(3.0)},
        std::tuple{Scalar(0.5), Scalar(-1.0)});

    constexpr Index dim0 = 2;
    constexpr Index dim1 = 3;

    NNGraph g("add_structure");
    auto* x = g.tensor({dim0, dim1}, "x", DataType::FP32);
    auto* y = g.tensor({dim0, dim1}, "y", DataType::FP32);
    auto* z = add(alpha, x, beta, y, "z");

    REQUIRE(z != nullptr);
    REQUIRE(z->has_producer());
    REQUIRE(z->shape() == (std::vector<Index>{dim0, dim1}));
    REQUIRE(g.num_ops() == 1);
    REQUIRE(g.tensor_graph().ops()[0]->op_name() == "ADD");
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph add backward", "[graph][nn_graph]")
{
    const auto [alpha, beta, grad_fill_val] = GENERATE(
        std::tuple{Scalar(2.0), Scalar(3.0), Scalar(1.0)},
        std::tuple{Scalar(0.5), Scalar(-1.0), Scalar(2.0)},
        std::tuple{Scalar(1.0), Scalar(0.0), Scalar(1.0)});

    NNGraph g("autograd_add");
    auto* x = g.tensor({2, 3}, "x", DataType::FP32);
    auto* y = g.tensor({2, 3}, "y", DataType::FP32);

    auto* z = add(alpha, x, beta, y, "z");

    REQUIRE(z->has_producer());
    REQUIRE_FALSE(z->is_leaf());
    REQUIRE(x->is_leaf());
    REQUIRE(y->is_leaf());

    auto [z_grad, _] = g.get_or_create_grad(z, "z_grad");
    gt::fill(grad_fill_val, z_grad->data());
    z->backward();

    REQUIRE(x->has_grad());
    REQUIRE(y->has_grad());

    size_t add_inplace_count = 0;
    for(const auto& op : g.tensor_graph().ops())
    {
        if(op->op_name() == "ADD_INPLACE")
            ++add_inplace_count;
    }
    REQUIRE(add_inplace_count == 2);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph add chain", "[graph][nn_graph]")
{
    const auto [add_alpha, add_beta, grad_fill_val] = GENERATE(
        std::tuple{Scalar(1.0), Scalar(1.0), Scalar(1.0)},
        std::tuple{Scalar(0.5), Scalar(2.0), Scalar(-1.0)});

    NNGraph g("add_chain");
    auto* x = g.tensor({2, 2}, "x", DataType::FP32);
    auto* y = g.tensor({2, 2}, "y", DataType::FP32);
    auto* u = g.tensor({2, 2}, "u", DataType::FP32);

    auto* w = add(add_alpha, x, add_beta, y, "w");
    auto* z = add(add_alpha, w, add_beta, u, "z");

    auto [z_grad, _] = g.get_or_create_grad(z, "z_grad");
    gt::fill(grad_fill_val, z_grad->data());
    z->backward();

    REQUIRE(x->has_grad());
    REQUIRE(y->has_grad());
    REQUIRE(u->has_grad());
    REQUIRE(w->has_grad());
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph add diamond", "[graph][nn_graph]")
{
    const auto [add_alpha, add_beta, grad_fill_val] = GENERATE(
        std::tuple{Scalar(1.0), Scalar(1.0), Scalar(1.0)},
        std::tuple{Scalar(2.0), Scalar(0.5), Scalar(1.0)});

    NNGraph g("add_diamond");
    auto* x = g.tensor({2, 2}, "x", DataType::FP32);
    auto* y = g.tensor({2, 2}, "y", DataType::FP32);

    auto* w = add(add_alpha, x, add_beta, y, "w");
    auto* v = add(add_alpha, w, add_beta, y, "v");
    auto* z = add(add_alpha, v, add_beta, w, "z");

    auto [z_grad, _] = g.get_or_create_grad(z, "z_grad");
    gt::fill(grad_fill_val, z_grad->data());
    z->backward();

    REQUIRE(x->has_grad());
    REQUIRE(y->has_grad());
    REQUIRE(w->has_grad());
    REQUIRE(v->has_grad());
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph add forward and backward", "[graph][nn_graph]")
{
    const auto [add_alpha, add_beta, grad_fill_val] = GENERATE(
        std::tuple{Scalar(1.0), Scalar(1.0), Scalar(1.0)},
        std::tuple{Scalar(2.0), Scalar(3.0), Scalar(1.0)},
        std::tuple{Scalar(0.5), Scalar(-1.0), Scalar(2.0)});

    NNGraph g("add");
    auto* x = g.tensor({2, 3}, "x", DataType::FP32, true);
    auto* y = g.tensor({2, 3}, "y", DataType::FP32, true);
    auto* z = add(add_alpha, x, add_beta, y, "z");

    REQUIRE(z != nullptr);
    REQUIRE(z->has_producer());
    REQUIRE(z->shape() == (std::vector<Index>{2, 3}));

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
    "NNGraph add forward matches PyTorch", "[graph][nn_graph][pytorch]")
{
    const auto [alpha, beta] = GENERATE(
        std::tuple{Scalar(1.0), Scalar(1.0)},
        std::tuple{Scalar(2.0), Scalar(3.0)},
        std::tuple{Scalar(0.5), Scalar(-1.0)});

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

    NNGraph g("add_pytorch");
    auto* x = g.tensor({dim0, dim1}, "x", DataType::FP32, true);
    auto* y = g.tensor({dim0, dim1}, "y", DataType::FP32, true);
    auto* z = add(alpha, x, beta, y, "z");

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
                    .clone()
                    .set_requires_grad(false);
    auto y_pt = torch::from_blob(y_data.data(), {dim0, dim1},
                                 torch::TensorOptions().dtype(torch::kFloat32))
                    .clone()
                    .set_requires_grad(false);

    auto z_pt = x_pt.mul(alpha).add(y_pt, beta);
    std::vector<float> pytorch_out(z_pt.data_ptr<float>(),
                                   z_pt.data_ptr<float>() + nelems);

    REQUIRE(nntile_out.size() == pytorch_out.size());
    for(size_t i = 0; i < nntile_out.size(); ++i)
    {
        REQUIRE(std::abs(nntile_out[i] - pytorch_out[i]) < pytorch_tolerance);
    }
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph add backward matches PyTorch", "[graph][nn_graph][pytorch]")
{
    const auto [alpha, beta, grad_fill_val] = GENERATE(
        std::tuple{Scalar(1.0), Scalar(1.0), Scalar(1.0)},
        std::tuple{Scalar(2.0), Scalar(3.0), Scalar(1.0)},
        std::tuple{Scalar(0.5), Scalar(-1.0), Scalar(2.0)});

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

    NNGraph g("add_bwd_pytorch");
    auto* x = g.tensor({dim0, dim1}, "x", DataType::FP32, true);
    auto* y = g.tensor({dim0, dim1}, "y", DataType::FP32, true);
    auto* z = add(alpha, x, beta, y, "z");

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
                    .clone()
                    .set_requires_grad(true);
    auto y_pt = torch::from_blob(y_data.data(), {dim0, dim1},
                                 torch::TensorOptions().dtype(torch::kFloat32))
                    .clone()
                    .set_requires_grad(true);

    auto z_pt = x_pt.mul(alpha).add(y_pt, beta);
    auto grad_output = torch::full(
        {dim0, dim1}, static_cast<float>(grad_fill_val),
        torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false));
    z_pt.backward(grad_output);

    compare_float_vectors(nntile_grad_x, x_pt.grad());
    compare_float_vectors(nntile_grad_y, y_pt.grad());
}

#endif // NNTILE_HAVE_TORCH
