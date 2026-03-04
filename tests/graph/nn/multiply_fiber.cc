/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/nn/multiply_fiber.cc
 * Test NNGraph multiply_fiber autograd operation.
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
constexpr Index dim_4 = 4;

} // anonymous namespace

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph multiply_fiber structure", "[graph][nn_graph]")
{
    const auto [alpha, axis] = GENERATE(
        std::tuple{Scalar(1.0), Index(0)},
        std::tuple{Scalar(1.0), Index(1)},
        std::tuple{Scalar(2.0), Index(0)},
        std::tuple{Scalar(0.5), Index(1)});

    std::vector<Index> fiber_shape = (axis == 0) ?
        std::vector<Index>{dim_2} : std::vector<Index>{dim_4};
    NNGraph g("multiply_fiber_structure");
    auto* src1 = g.tensor(fiber_shape, "src1", DataType::FP32);
    auto* src2 = g.tensor({dim_2, dim_4}, "src2", DataType::FP32);
    auto* out = multiply_fiber(alpha, src1, src2, "out", axis);

    REQUIRE(out != nullptr);
    REQUIRE(out->has_producer());
    REQUIRE(out->shape() == (std::vector<Index>{dim_2, dim_4}));
    REQUIRE(g.num_ops() == 1);
    REQUIRE(g.tensor_graph().ops()[0]->op_name() == "MULTIPLY_FIBER");
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph multiply_fiber backward", "[graph][nn_graph]")
{
    const auto [alpha, axis, grad_fill_val] = GENERATE(
        std::tuple{Scalar(1.0), Index(1), Scalar(1.0)},
        std::tuple{Scalar(1.0), Index(0), Scalar(-1.0)},
        std::tuple{Scalar(2.0), Index(0), Scalar(0.5)},
        std::tuple{Scalar(0.5), Index(1), Scalar(2.0)});

    std::vector<Index> fiber_shape = (axis == 0) ?
        std::vector<Index>{dim_2} : std::vector<Index>{dim_4};
    NNGraph g("multiply_fiber_backward");
    auto* src1 = g.tensor(fiber_shape, "src1", DataType::FP32);
    auto* src2 = g.tensor({dim_2, dim_4}, "src2", DataType::FP32);
    auto* out = multiply_fiber(alpha, src1, src2, "out", axis);

    auto [out_grad, _] = g.get_or_create_grad(out, "out_grad");
    gt::fill(grad_fill_val, out_grad->data());
    out->backward();

    REQUIRE(src1->has_grad());
    REQUIRE(src2->has_grad());
}

#ifdef NNTILE_HAVE_TORCH

using nntile::test::broadcast_fiber;
using nntile::test::compare_float_vectors;
using nntile::test::pytorch_tolerance;

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph multiply_fiber forward matches PyTorch", "[graph][nn_graph][pytorch]")
{
    const auto [alpha, axis] = GENERATE(
        std::tuple{Scalar(1.0), Index(0)},
        std::tuple{Scalar(1.0), Index(1)},
        std::tuple{Scalar(2.0), Index(0)},
        std::tuple{Scalar(0.5), Index(1)});

    std::vector<Index> fiber_shape = (axis == 0) ?
        std::vector<Index>{dim_2} : std::vector<Index>{dim_4};
    const Index fiber_nelems = static_cast<Index>(fiber_shape[0]);
    constexpr Index nelems = dim_2 * dim_4;

    std::vector<float> src1_data(fiber_nelems);
    std::vector<float> src2_data(nelems);
    for(Index i = 0; i < fiber_nelems; ++i)
        src1_data[i] = 0.1f * static_cast<float>(i + 1);
    for(Index i = 0; i < nelems; ++i)
        src2_data[i] = 0.2f * static_cast<float>(-i - 1);

    NNGraph g("multiply_fiber_pytorch");
    auto* src1 = g.tensor(fiber_shape, "src1", DataType::FP32, true);
    auto* src2 = g.tensor({dim_2, dim_4}, "src2", DataType::FP32, true);
    auto* out = multiply_fiber(alpha, src1, src2, "out", axis);

    src1->mark_input(true);
    src2->mark_input(true);
    out->mark_output(true);

    TensorGraph::Runtime runtime(g.tensor_graph());
    runtime.compile();
    runtime.bind_data("src1", src1_data);
    runtime.bind_data("src2", src2_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> nntile_out = runtime.get_output<float>("out");

    auto src1_pt = torch::from_blob(src1_data.data(),
        {static_cast<long>(fiber_nelems)},
        torch::TensorOptions().dtype(torch::kFloat32)).clone().set_requires_grad(false);
    auto src2_pt = torch::from_blob(src2_data.data(), {dim_2, dim_4},
        torch::TensorOptions().dtype(torch::kFloat32)).clone().set_requires_grad(false);

    auto src1_bc = broadcast_fiber(src1_pt, {dim_2, dim_4}, axis);
    auto out_pt = (alpha * src1_bc * src2_pt).contiguous();
    std::vector<float> pytorch_out(out_pt.data_ptr<float>(),
                                   out_pt.data_ptr<float>() + nelems);

    REQUIRE(nntile_out.size() == pytorch_out.size());
    for(size_t i = 0; i < nntile_out.size(); ++i)
        REQUIRE(std::abs(nntile_out[i] - pytorch_out[i]) < pytorch_tolerance);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph multiply_fiber backward matches PyTorch", "[graph][nn_graph][pytorch]")
{
    const auto [alpha, axis, grad_fill_val] = GENERATE(
        std::tuple{Scalar(1.0), Index(1), Scalar(1.0)},
        std::tuple{Scalar(1.0), Index(0), Scalar(-1.0)},
        std::tuple{Scalar(2.0), Index(0), Scalar(0.5)},
        std::tuple{Scalar(0.5), Index(1), Scalar(2.0)});

    std::vector<Index> fiber_shape = (axis == 0) ?
        std::vector<Index>{dim_2} : std::vector<Index>{dim_4};
    const Index fiber_nelems = static_cast<Index>(fiber_shape[0]);
    constexpr Index nelems = dim_2 * dim_4;

    std::vector<float> src1_data(fiber_nelems);
    std::vector<float> src2_data(nelems);
    for(Index i = 0; i < fiber_nelems; ++i)
        src1_data[i] = 0.1f * static_cast<float>(i);
    for(Index i = 0; i < nelems; ++i)
        src2_data[i] = 0.15f * static_cast<float>(i + 10);

    NNGraph g("multiply_fiber_bwd_pytorch");
    auto* src1 = g.tensor(fiber_shape, "src1", DataType::FP32, true);
    auto* src2 = g.tensor({dim_2, dim_4}, "src2", DataType::FP32, true);
    auto* out = multiply_fiber(alpha, src1, src2, "out", axis);

    src1->mark_input(true);
    src2->mark_input(true);

    auto [out_grad, _] = g.get_or_create_grad(out, "out_grad");
    gt::fill(grad_fill_val, out_grad->data());
    out->backward();

    src1->grad()->mark_output(true);
    src2->grad()->mark_output(true);

    TensorGraph::Runtime runtime(g.tensor_graph());
    runtime.compile();
    runtime.bind_data("src1", src1_data);
    runtime.bind_data("src2", src2_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> nntile_grad_src1 =
        runtime.get_output<float>(src1->grad()->name());
    std::vector<float> nntile_grad_src2 =
        runtime.get_output<float>(src2->grad()->name());

    auto src1_pt = torch::from_blob(src1_data.data(),
        {static_cast<long>(fiber_nelems)},
        torch::TensorOptions().dtype(torch::kFloat32)).clone().set_requires_grad(true);
    auto src2_pt = torch::from_blob(src2_data.data(), {dim_2, dim_4},
        torch::TensorOptions().dtype(torch::kFloat32)).clone().set_requires_grad(true);

    auto src1_bc = broadcast_fiber(src1_pt, {dim_2, dim_4}, axis);
    auto out_pt = alpha * src1_bc * src2_pt;
    auto grad_output = torch::full(
        {dim_2, dim_4}, static_cast<float>(grad_fill_val),
        torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false));
    out_pt.backward(grad_output);

    compare_float_vectors(nntile_grad_src1, src1_pt.grad());
    compare_float_vectors(nntile_grad_src2, src2_pt.grad());
}

#endif // NNTILE_HAVE_TORCH
