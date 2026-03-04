/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/nn/scale_fiber.cc
 * Test NNGraph scale_fiber autograd operation.
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

constexpr Index batch_ndim_none = 0;
constexpr Index dim_2 = 2;
constexpr Index dim_4 = 4;

} // anonymous namespace

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph scale_fiber structure", "[graph][nn_graph]")
{
    const auto [alpha, axis] = GENERATE(
        std::tuple{Scalar(1.0), Index(0)},
        std::tuple{Scalar(2.0), Index(1)},
        std::tuple{Scalar(0.5), Index(0)},
        std::tuple{Scalar(-1.0), Index(1)});

    std::vector<Index> fiber_shape = (axis == 0) ?
        std::vector<Index>{dim_2} : std::vector<Index>{dim_4};
    std::vector<Index> dst_shape = {dim_2, dim_4};

    NNGraph g("scale_fiber_structure");
    auto* src = g.tensor(fiber_shape, "src", DataType::FP32);
    auto* out = scale_fiber(alpha, src, "out", dst_shape, axis, batch_ndim_none);

    REQUIRE(out != nullptr);
    REQUIRE(out->has_producer());
    REQUIRE(out->shape() == dst_shape);
    REQUIRE(g.num_ops() == 1);
    REQUIRE(g.tensor_graph().ops()[0]->op_name() == "SCALE_FIBER");
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph scale_fiber backward", "[graph][nn_graph]")
{
    const auto [alpha, axis, grad_fill_val] = GENERATE(
        std::tuple{Scalar(2.0), Index(1), Scalar(1.0)},
        std::tuple{Scalar(0.5), Index(0), Scalar(-1.0)},
        std::tuple{Scalar(1.0), Index(0), Scalar(0.5)},
        std::tuple{Scalar(-1.0), Index(1), Scalar(2.0)});

    std::vector<Index> fiber_shape = (axis == 0) ?
        std::vector<Index>{dim_2} : std::vector<Index>{dim_4};
    std::vector<Index> dst_shape = {dim_2, dim_4};

    NNGraph g("scale_fiber_backward");
    auto* src = g.tensor(fiber_shape, "src", DataType::FP32);
    auto* out = scale_fiber(alpha, src, "out", dst_shape, axis, batch_ndim_none);

    auto [out_grad, _] = g.get_or_create_grad(out, "out_grad");
    gt::fill(grad_fill_val, out_grad->data());
    out->backward();

    REQUIRE(src->has_grad());
    REQUIRE(src->grad()->shape() == fiber_shape);
}

#ifdef NNTILE_HAVE_TORCH

using nntile::test::broadcast_fiber;
using nntile::test::compare_float_vectors;
using nntile::test::colmajor_to_rowmajor;
using nntile::test::pytorch_tolerance;

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph scale_fiber forward matches PyTorch", "[graph][nn_graph][pytorch]")
{
    const auto [alpha, axis] = GENERATE(
        std::tuple{Scalar(1.0), Index(1)},
        std::tuple{Scalar(2.0), Index(0)},
        std::tuple{Scalar(0.5), Index(0)},
        std::tuple{Scalar(-1.0), Index(1)});

    std::vector<Index> fiber_shape = (axis == 0) ?
        std::vector<Index>{dim_2} : std::vector<Index>{dim_4};
    std::vector<Index> dst_shape = {dim_2, dim_4};
    const Index fiber_nelems = static_cast<Index>(fiber_shape[0]);
    const Index dst_nelems = dim_2 * dim_4;

    std::vector<float> src_data(fiber_nelems);
    for(Index i = 0; i < fiber_nelems; ++i)
        src_data[i] = 0.1f * static_cast<float>(i + 1);

    NNGraph g("scale_fiber_pytorch");
    auto* src = g.tensor(fiber_shape, "src", DataType::FP32, true);
    auto* out = scale_fiber(alpha, src, "out", dst_shape, axis, batch_ndim_none);

    src->mark_input(true);
    out->mark_output(true);

    TensorGraph::Runtime runtime(g.tensor_graph());
    runtime.compile();
    runtime.bind_data("src", src_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> nntile_out_colmajor = runtime.get_output<float>("out");
    std::vector<float> nntile_out =
        colmajor_to_rowmajor(nntile_out_colmajor, dst_shape);

    std::vector<::int64_t> dst_shape_pt(dst_shape.begin(), dst_shape.end());
    auto src_pt = torch::from_blob(src_data.data(),
        {static_cast<long>(fiber_nelems)},
        torch::TensorOptions().dtype(torch::kFloat32)).clone().set_requires_grad(false);
    auto src_bc = broadcast_fiber(src_pt, dst_shape_pt, axis);
    auto out_pt = (alpha * src_bc).contiguous();

    std::vector<float> pytorch_out(out_pt.data_ptr<float>(),
                                   out_pt.data_ptr<float>() + dst_nelems);

    REQUIRE(nntile_out.size() == pytorch_out.size());
    for(size_t i = 0; i < nntile_out.size(); ++i)
        REQUIRE(std::abs(nntile_out[i] - pytorch_out[i]) < pytorch_tolerance);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph scale_fiber backward matches PyTorch", "[graph][nn_graph][pytorch]")
{
    const auto [alpha, axis, grad_fill_val] = GENERATE(
        std::tuple{Scalar(2.0), Index(1), Scalar(1.0)},
        std::tuple{Scalar(0.5), Index(0), Scalar(-1.0)},
        std::tuple{Scalar(1.0), Index(0), Scalar(0.5)},
        std::tuple{Scalar(-1.0), Index(1), Scalar(2.0)});

    std::vector<Index> fiber_shape = (axis == 0) ?
        std::vector<Index>{dim_2} : std::vector<Index>{dim_4};
    std::vector<Index> dst_shape = {dim_2, dim_4};
    const Index fiber_nelems = static_cast<Index>(fiber_shape[0]);

    std::vector<float> src_data(fiber_nelems);
    for(Index i = 0; i < fiber_nelems; ++i)
        src_data[i] = 0.1f * static_cast<float>(i);

    NNGraph g("scale_fiber_bwd_pytorch");
    auto* src = g.tensor(fiber_shape, "src", DataType::FP32, true);
    auto* out = scale_fiber(alpha, src, "out", dst_shape, axis, batch_ndim_none);

    src->mark_input(true);

    auto [out_grad, _] = g.get_or_create_grad(out, "out_grad");
    gt::fill(grad_fill_val, out_grad->data());
    out->backward();

    src->grad()->mark_output(true);

    TensorGraph::Runtime runtime(g.tensor_graph());
    runtime.compile();
    runtime.bind_data("src", src_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> nntile_grad_src =
        runtime.get_output<float>(src->grad()->name());

    std::vector<::int64_t> dst_shape_pt(dst_shape.begin(), dst_shape.end());
    auto src_pt = torch::from_blob(src_data.data(),
        {static_cast<long>(fiber_nelems)},
        torch::TensorOptions().dtype(torch::kFloat32)).clone().set_requires_grad(true);
    auto src_bc = broadcast_fiber(src_pt, dst_shape_pt, axis);
    auto out_pt = alpha * src_bc;

    auto grad_output = torch::full(dst_shape_pt,
        static_cast<float>(grad_fill_val),
        torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false));
    out_pt.backward(grad_output);

    compare_float_vectors(nntile_grad_src, src_pt.grad());
}

#endif // NNTILE_HAVE_TORCH
