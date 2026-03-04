/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/nn/sum_slice.cc
 * Test NNGraph sum_slice autograd operation.
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
constexpr Index dim_4 = 4;
constexpr int redux_none = 0;

} // anonymous namespace

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph sum_slice structure", "[graph][nn_graph]")
{
    const auto [alpha, axis] = GENERATE(
        std::tuple{Scalar(1.0), Index(0)},
        std::tuple{Scalar(1.0), Index(1)},
        std::tuple{Scalar(2.0), Index(0)},
        std::tuple{Scalar(0.5), Index(1)});

    std::vector<Index> x_shape = (axis == 0) ?
        std::vector<Index>{dim_4, dim_2, dim_3} : std::vector<Index>{dim_2, dim_4, dim_3};
    std::vector<Index> out_shape;
    for(Index i = 0; i < static_cast<Index>(x_shape.size()); ++i)
        if(i != axis)
            out_shape.push_back(x_shape[i]);

    NNGraph g("sum_slice_structure");
    auto* x = g.tensor(x_shape, "x", DataType::FP32);
    auto* y = sum_slice(x, "y", axis, redux_none, alpha);

    REQUIRE(y != nullptr);
    REQUIRE(y->has_producer());
    REQUIRE(y->shape() == out_shape);
    REQUIRE(g.num_ops() == 1);
    REQUIRE(g.tensor_graph().ops()[0]->op_name() == "SUM_SLICE");
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph sum_slice backward", "[graph][nn_graph]")
{
    const auto [alpha, axis, grad_fill_val] = GENERATE(
        std::tuple{Scalar(1.0), Index(1), Scalar(1.0)},
        std::tuple{Scalar(1.0), Index(0), Scalar(-1.0)},
        std::tuple{Scalar(2.0), Index(0), Scalar(0.5)},
        std::tuple{Scalar(0.5), Index(1), Scalar(2.0)});

    std::vector<Index> x_shape = (axis == 0) ?
        std::vector<Index>{dim_4, dim_2} : std::vector<Index>{dim_2, dim_4};

    NNGraph g("sum_slice_backward");
    auto* x = g.tensor(x_shape, "x", DataType::FP32);
    auto* y = sum_slice(x, "y", axis, redux_none, alpha);

    auto [y_grad, _] = g.get_or_create_grad(y, "y_grad");
    gt::fill(grad_fill_val, y_grad->data());
    y->backward();

    REQUIRE(x->has_grad());
    REQUIRE(x->grad()->shape() == x_shape);
}

#ifdef NNTILE_HAVE_TORCH

using nntile::test::compare_float_vectors;
using nntile::test::colmajor_to_rowmajor;
using nntile::test::pytorch_tolerance;

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph sum_slice forward matches PyTorch", "[graph][nn_graph][pytorch]")
{
    const auto [alpha, axis] = GENERATE(
        std::tuple{Scalar(1.0), Index(0)},
        std::tuple{Scalar(1.0), Index(1)},
        std::tuple{Scalar(2.0), Index(0)},
        std::tuple{Scalar(0.5), Index(1)});

    std::vector<Index> x_shape = (axis == 0) ?
        std::vector<Index>{dim_4, dim_2} : std::vector<Index>{dim_2, dim_4};
    const Index x_nelems = dim_2 * dim_4;

    std::vector<float> x_data(x_nelems);
    for(Index i = 0; i < x_nelems; ++i)
        x_data[i] = 0.1f * static_cast<float>(i + 1);
    std::vector<float> x_rowmajor = colmajor_to_rowmajor(x_data, x_shape);

    NNGraph g("sum_slice_pytorch");
    auto* x = g.tensor(x_shape, "x", DataType::FP32, true);
    auto* y = sum_slice(x, "y", axis, redux_none, alpha);

    x->mark_input(true);
    y->mark_output(true);

    TensorGraph::Runtime runtime(g.tensor_graph());
    runtime.compile();
    runtime.bind_data("x", x_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> nntile_out = runtime.get_output<float>("y");

    std::vector<::int64_t> x_shape_pt(x_shape.begin(), x_shape.end());
    auto x_pt = torch::from_blob(x_rowmajor.data(), x_shape_pt,
        torch::TensorOptions().dtype(torch::kFloat32)).clone().set_requires_grad(false);
    auto y_pt = (alpha * x_pt.sum(static_cast<int64_t>(axis), false)).contiguous();

    std::vector<float> pytorch_out(y_pt.data_ptr<float>(),
                                   y_pt.data_ptr<float>() + y_pt.numel());

    REQUIRE(nntile_out.size() == pytorch_out.size());
    for(size_t i = 0; i < nntile_out.size(); ++i)
        REQUIRE(std::abs(nntile_out[i] - pytorch_out[i]) < pytorch_tolerance);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph sum_slice backward matches PyTorch", "[graph][nn_graph][pytorch]")
{
    const auto [alpha, axis, grad_fill_val] = GENERATE(
        std::tuple{Scalar(1.0), Index(1), Scalar(1.0)},
        std::tuple{Scalar(1.0), Index(0), Scalar(-1.0)},
        std::tuple{Scalar(2.0), Index(0), Scalar(0.5)},
        std::tuple{Scalar(0.5), Index(1), Scalar(2.0)});

    std::vector<Index> x_shape = (axis == 0) ?
        std::vector<Index>{dim_4, dim_2} : std::vector<Index>{dim_2, dim_4};
    const Index x_nelems = dim_2 * dim_4;

    std::vector<float> x_data(x_nelems);
    for(Index i = 0; i < x_nelems; ++i)
        x_data[i] = 0.1f * static_cast<float>(i);
    std::vector<float> x_rowmajor = colmajor_to_rowmajor(x_data, x_shape);

    NNGraph g("sum_slice_bwd_pytorch");
    auto* x = g.tensor(x_shape, "x", DataType::FP32, true);
    auto* y = sum_slice(x, "y", axis, redux_none, alpha);

    x->mark_input(true);

    auto [y_grad, _] = g.get_or_create_grad(y, "y_grad");
    gt::fill(grad_fill_val, y_grad->data());
    y->backward();

    x->grad()->mark_output(true);

    TensorGraph::Runtime runtime(g.tensor_graph());
    runtime.compile();
    runtime.bind_data("x", x_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> nntile_grad_x_colmajor =
        runtime.get_output<float>(x->grad()->name());
    std::vector<float> nntile_grad_x =
        colmajor_to_rowmajor(nntile_grad_x_colmajor, x_shape);

    std::vector<::int64_t> x_shape_pt(x_shape.begin(), x_shape.end());
    auto x_pt = torch::from_blob(x_rowmajor.data(), x_shape_pt,
        torch::TensorOptions().dtype(torch::kFloat32)).clone().set_requires_grad(true);
    auto y_pt = alpha * x_pt.sum(static_cast<int64_t>(axis), false);

    auto grad_y = torch::full({y_pt.numel()}, static_cast<float>(grad_fill_val),
        torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false));
    y_pt.backward(grad_y);

    compare_float_vectors(nntile_grad_x, x_pt.grad());
}

#endif // NNTILE_HAVE_TORCH
