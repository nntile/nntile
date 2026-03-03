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

#ifdef NNTILE_HAVE_TORCH
#   include "pytorch_helper.hh"
#endif

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

    auto [out_grad, _] = g.get_or_create_grad(out, "out_grad");
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

    auto [out_grad, _] = g.get_or_create_grad(out, "out_grad");
    fill(grad_fill_val, out_grad->data());
    out->backward();

    REQUIRE(fiber->has_grad());
    REQUIRE(tensor->has_grad());
    REQUIRE(fiber->grad()->shape() == fiber_shape);
    REQUIRE(tensor->grad()->shape() == tensor_shape);
}

#ifdef NNTILE_HAVE_TORCH

using nntile::test::broadcast_fiber;
using nntile::test::colmajor_to_rowmajor;
using nntile::test::compare_float_vectors;
using nntile::test::pytorch_tolerance;

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph add_fiber forward matches PyTorch", "[graph][nn_graph][pytorch]")
{
    const auto [alpha, beta, axis] = GENERATE(
        std::tuple{Scalar(1.0), Scalar(1.0), Index(1)},
        std::tuple{Scalar(0.5), Scalar(2.0), Index(0)},
        std::tuple{Scalar(2.0), Scalar(0.0), Index(1)});

    constexpr Index batch_ndim_none = 0;
    constexpr Index dim0 = 2;
    constexpr Index dim1 = 4;
    std::vector<Index> tensor_shape = (axis == 0) ?
        std::vector<Index>{dim1, dim0} : std::vector<Index>{dim0, dim1};
    std::vector<Index> fiber_shape_vec = {tensor_shape[axis]};

    const Index tensor_nelems = dim0 * dim1;
    const Index fiber_nelems = tensor_shape[axis];

    // Same data pattern as tests/graph/tensor/add_fiber.cc (column-major for NNTile)
    std::vector<float> fiber_data(fiber_nelems);
    std::vector<float> tensor_data(tensor_nelems);
    for(Index i = 0; i < fiber_nelems; ++i)
        fiber_data[i] = static_cast<float>(i + 1);
    for(Index i = 0; i < tensor_nelems; ++i)
        tensor_data[i] = static_cast<float>(-i - 1);
    std::vector<float> tensor_data_rowmajor =
        colmajor_to_rowmajor(tensor_data, tensor_shape);

    NNGraph g("add_fiber_pytorch");
    auto* fiber = g.tensor(fiber_shape_vec, "fiber", DataType::FP32, true);
    auto* tensor = g.tensor(tensor_shape, "tensor", DataType::FP32, true);
    auto* out = add_fiber(alpha, fiber, beta, tensor, "out",
                         axis, batch_ndim_none);

    fiber->mark_input(true);
    tensor->mark_input(true);
    out->mark_output(true);

    TensorGraph::Runtime runtime(g.tensor_graph());
    runtime.compile();
    runtime.bind_data("fiber", fiber_data);
    runtime.bind_data("tensor", tensor_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> nntile_out_colmajor = runtime.get_output<float>("out");
    std::vector<float> nntile_out =
        colmajor_to_rowmajor(nntile_out_colmajor, tensor_shape);

    // PyTorch: output = alpha * fiber (broadcast) + beta * tensor (row-major)
    std::vector<::int64_t> tensor_shape_pt(tensor_shape.begin(), tensor_shape.end());
    auto fiber_pt = torch::from_blob(fiber_data.data(),
        {static_cast<long>(fiber_nelems)},
        torch::TensorOptions().dtype(torch::kFloat32)).clone().set_requires_grad(false);
    auto tensor_pt = torch::from_blob(tensor_data_rowmajor.data(), tensor_shape_pt,
        torch::TensorOptions().dtype(torch::kFloat32)).clone().set_requires_grad(false);

    auto fiber_bc = broadcast_fiber(fiber_pt, tensor_shape_pt, axis);
    auto out_pt = (alpha * fiber_bc + beta * tensor_pt).contiguous();

    std::vector<float> pytorch_out(out_pt.data_ptr<float>(),
                                   out_pt.data_ptr<float>() + tensor_nelems);

    REQUIRE(nntile_out.size() == pytorch_out.size());
    for(size_t i = 0; i < nntile_out.size(); ++i)
        REQUIRE(std::abs(nntile_out[i] - pytorch_out[i]) < pytorch_tolerance);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph add_fiber backward matches PyTorch", "[graph][nn_graph][pytorch]")
{
    const auto [alpha, beta, axis, grad_fill_val] = GENERATE(
        std::tuple{Scalar(1.0), Scalar(1.0), Index(1), Scalar(1.0)},
        std::tuple{Scalar(2.0), Scalar(0.0), Index(0), Scalar(-1.0)},
        std::tuple{Scalar(0.5), Scalar(2.0), Index(1), Scalar(1.0)});

    constexpr Index batch_ndim_none = 0;
    constexpr Index dim0 = 2;
    constexpr Index dim1 = 4;
    std::vector<Index> tensor_shape = (axis == 0) ?
        std::vector<Index>{dim1, dim0} : std::vector<Index>{dim0, dim1};
    std::vector<Index> fiber_shape_vec = {tensor_shape[axis]};

    const Index tensor_nelems = dim0 * dim1;
    const Index fiber_nelems = tensor_shape[axis];

    std::vector<float> fiber_data(fiber_nelems);
    std::vector<float> tensor_data(tensor_nelems);
    for(Index i = 0; i < fiber_nelems; ++i)
        fiber_data[i] = 0.1f * static_cast<float>(i);
    for(Index i = 0; i < tensor_nelems; ++i)
        tensor_data[i] = 0.15f * static_cast<float>(i + 5);
    std::vector<float> tensor_data_rowmajor =
        colmajor_to_rowmajor(tensor_data, tensor_shape);

    NNGraph g("add_fiber_bwd_pytorch");
    auto* fiber = g.tensor(fiber_shape_vec, "fiber", DataType::FP32, true);
    auto* tensor = g.tensor(tensor_shape, "tensor", DataType::FP32, true);
    auto* out = add_fiber(alpha, fiber, beta, tensor, "out",
                         axis, batch_ndim_none);

    fiber->mark_input(true);
    tensor->mark_input(true);

    auto [out_grad, _] = g.get_or_create_grad(out, "out_grad");
    fill(grad_fill_val, out_grad->data());
    out->backward();

    fiber->grad()->mark_output(true);
    tensor->grad()->mark_output(true);

    TensorGraph::Runtime runtime(g.tensor_graph());
    runtime.compile();
    runtime.bind_data("fiber", fiber_data);
    runtime.bind_data("tensor", tensor_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> nntile_grad_fiber =
        runtime.get_output<float>(fiber->grad()->name());
    std::vector<float> nntile_grad_tensor_colmajor =
        runtime.get_output<float>(tensor->grad()->name());
    std::vector<float> nntile_grad_tensor =
        colmajor_to_rowmajor(nntile_grad_tensor_colmajor, tensor_shape);

    std::vector<::int64_t> tensor_shape_pt(tensor_shape.begin(), tensor_shape.end());
    auto fiber_pt = torch::from_blob(fiber_data.data(),
        {static_cast<long>(fiber_nelems)},
        torch::TensorOptions().dtype(torch::kFloat32)).clone().set_requires_grad(true);
    auto tensor_pt = torch::from_blob(tensor_data_rowmajor.data(), tensor_shape_pt,
        torch::TensorOptions().dtype(torch::kFloat32)).clone().set_requires_grad(true);

    auto fiber_bc = broadcast_fiber(fiber_pt, tensor_shape_pt, axis);
    auto out_pt = alpha * fiber_bc + beta * tensor_pt;

    auto grad_output = torch::full(tensor_shape_pt,
        static_cast<float>(grad_fill_val),
        torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false));
    out_pt.backward(grad_output);

    compare_float_vectors(nntile_grad_fiber, fiber_pt.grad());
    compare_float_vectors(nntile_grad_tensor, tensor_pt.grad());
}

#endif // NNTILE_HAVE_TORCH
