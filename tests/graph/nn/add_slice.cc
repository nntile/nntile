/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/nn/add_slice.cc
 * Test NNGraph add_slice autograd operation.
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

static std::vector<Index> slice_shape(
    const std::vector<Index>& dst_shape,
    Index axis)
{
    std::vector<Index> out;
    out.reserve(dst_shape.size() - 1);
    for(Index i = 0; i < static_cast<Index>(dst_shape.size()); ++i)
    {
        if(i != axis)
        {
            out.push_back(dst_shape[i]);
        }
    }
    return out;
}

constexpr Index dim_2 = 2;
constexpr Index dim_3 = 3;
constexpr Index dim_4 = 4;

} // anonymous namespace

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph add_slice structure", "[graph][nn_graph]")
{
    const auto [alpha, beta, axis] = GENERATE(
        std::tuple{Scalar(1.0), Scalar(1.0), Index(1)},
        std::tuple{Scalar(0.5), Scalar(2.0), Index(0)},
        std::tuple{Scalar(0.0), Scalar(1.0), Index(0)},
        std::tuple{Scalar(2.5), Scalar(0.5), Index(1)},
        std::tuple{Scalar(0.25), Scalar(3.0), Index(0)},
        std::tuple{Scalar(1.0), Scalar(0.0), Index(1)});

    std::vector<Index> dst_shape = (axis == 0) ?
        std::vector<Index>{dim_4, dim_2, dim_3} : std::vector<Index>{dim_2, dim_4, dim_3};
    std::vector<Index> src1_shape = slice_shape(dst_shape, axis);

    NNGraph g("add_slice_structure");
    auto* src1 = g.tensor(src1_shape, "src1", DataType::FP32);
    auto* src2 = g.tensor(dst_shape, "src2", DataType::FP32);
    auto* out = add_slice(alpha, src1, beta, src2, "out", axis);

    REQUIRE(out != nullptr);
    REQUIRE(out->has_producer());
    REQUIRE(out->shape() == dst_shape);
    REQUIRE(g.num_ops() == 1);
    REQUIRE(g.tensor_graph().ops()[0]->op_name() == "ADD_SLICE");
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph add_slice backward", "[graph][nn_graph]")
{
    const auto [alpha, beta, axis, grad_fill_val] = GENERATE(
        std::tuple{Scalar(1.0), Scalar(1.0), Index(1), Scalar(1.0)},
        std::tuple{Scalar(2.0), Scalar(0.0), Index(0), Scalar(-1.0)},
        std::tuple{Scalar(0.5), Scalar(2.0), Index(0), Scalar(0.5)},
        std::tuple{Scalar(1.0), Scalar(0.0), Index(1), Scalar(2.0)},
        std::tuple{Scalar(0.25), Scalar(1.0), Index(0), Scalar(-2.0)});

    std::vector<Index> dst_shape = (axis == 0) ?
        std::vector<Index>{dim_4, dim_2} : std::vector<Index>{dim_2, dim_4};
    std::vector<Index> src1_shape = slice_shape(dst_shape, axis);

    NNGraph g("add_slice_backward");
    auto* src1 = g.tensor(src1_shape, "src1", DataType::FP32);
    auto* src2 = g.tensor(dst_shape, "src2", DataType::FP32);
    auto* out = add_slice(alpha, src1, beta, src2, "out", axis);

    auto [out_grad, _] = g.get_or_create_grad(out, "out_grad");
    gt::fill(grad_fill_val, out_grad->data());
    out->backward();

    REQUIRE(src1->has_grad());
    REQUIRE(src2->has_grad());
    REQUIRE(src1->grad()->shape() == src1_shape);
    REQUIRE(src2->grad()->shape() == dst_shape);
}

#ifdef NNTILE_HAVE_TORCH

using nntile::test::colmajor_to_rowmajor;
using nntile::test::compare_float_vectors;
using nntile::test::pytorch_tolerance;

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph add_slice forward matches PyTorch", "[graph][nn_graph][pytorch]")
{
    const auto [alpha, beta, axis] = GENERATE(
        std::tuple{Scalar(1.0), Scalar(1.0), Index(1)},
        std::tuple{Scalar(0.5), Scalar(2.0), Index(0)},
        std::tuple{Scalar(0.0), Scalar(1.0), Index(0)},
        std::tuple{Scalar(2.5), Scalar(0.5), Index(1)});

    std::vector<Index> dst_shape = (axis == 0) ?
        std::vector<Index>{dim_4, dim_2} : std::vector<Index>{dim_2, dim_4};
    std::vector<Index> src1_shape = slice_shape(dst_shape, axis);

    const Index dst_nelems = dim_2 * dim_4;
    Index src1_nelems = 1;
    for(Index d : src1_shape)
        src1_nelems *= d;

    std::vector<float> src1_data(static_cast<size_t>(src1_nelems));
    std::vector<float> src2_data(static_cast<size_t>(dst_nelems));
    for(Index i = 0; i < src1_nelems; ++i)
        src1_data[i] = static_cast<float>(i + 1);
    for(Index i = 0; i < dst_nelems; ++i)
        src2_data[i] = static_cast<float>(-i - 1);
    std::vector<float> src2_rowmajor = colmajor_to_rowmajor(src2_data, dst_shape);

    NNGraph g("add_slice_pytorch");
    auto* src1 = g.tensor(src1_shape, "src1", DataType::FP32, true);
    auto* src2 = g.tensor(dst_shape, "src2", DataType::FP32, true);
    auto* out = add_slice(alpha, src1, beta, src2, "out", axis);

    src1->mark_input(true);
    src2->mark_input(true);
    out->mark_output(true);

    TensorGraph::Runtime runtime(g.tensor_graph());
    runtime.compile();
    runtime.bind_data("src1", src1_data);
    runtime.bind_data("src2", src2_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> nntile_out_colmajor = runtime.get_output<float>("out");
    std::vector<float> nntile_out = colmajor_to_rowmajor(nntile_out_colmajor, dst_shape);

    std::vector<::int64_t> dst_shape_pt(dst_shape.begin(), dst_shape.end());
    std::vector<::int64_t> src1_shape_pt(src1_shape.begin(), src1_shape.end());
    auto src1_pt = torch::from_blob(src1_data.data(),
        src1_shape_pt,
        torch::TensorOptions().dtype(torch::kFloat32)).clone().set_requires_grad(false);
    auto src2_pt = torch::from_blob(src2_rowmajor.data(), dst_shape_pt,
        torch::TensorOptions().dtype(torch::kFloat32)).clone().set_requires_grad(false);

    auto src1_bc = src1_pt.unsqueeze(static_cast<int64_t>(axis))
                       .expand(dst_shape_pt);
    auto out_pt = (alpha * src1_bc + beta * src2_pt).contiguous();

    std::vector<float> pytorch_out(out_pt.data_ptr<float>(),
                                   out_pt.data_ptr<float>() + dst_nelems);

    REQUIRE(nntile_out.size() == pytorch_out.size());
    for(size_t i = 0; i < nntile_out.size(); ++i)
        REQUIRE(std::abs(nntile_out[i] - pytorch_out[i]) < pytorch_tolerance);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph add_slice backward matches PyTorch", "[graph][nn_graph][pytorch]")
{
    const auto [alpha, beta, axis, grad_fill_val] = GENERATE(
        std::tuple{Scalar(1.0), Scalar(1.0), Index(1), Scalar(1.0)},
        std::tuple{Scalar(2.0), Scalar(0.0), Index(0), Scalar(-1.0)},
        std::tuple{Scalar(0.5), Scalar(2.0), Index(0), Scalar(0.5)},
        std::tuple{Scalar(1.0), Scalar(0.0), Index(1), Scalar(2.0)});

    std::vector<Index> dst_shape = (axis == 0) ?
        std::vector<Index>{dim_4, dim_2} : std::vector<Index>{dim_2, dim_4};
    std::vector<Index> src1_shape = slice_shape(dst_shape, axis);

    const Index dst_nelems = dim_2 * dim_4;
    Index src1_nelems = 1;
    for(Index d : src1_shape)
        src1_nelems *= d;

    std::vector<float> src1_data(static_cast<size_t>(src1_nelems));
    std::vector<float> src2_data(dst_nelems);
    for(Index i = 0; i < src1_nelems; ++i)
        src1_data[i] = 0.1f * static_cast<float>(i);
    for(Index i = 0; i < dst_nelems; ++i)
        src2_data[i] = 0.15f * static_cast<float>(i + 5);
    std::vector<float> src2_rowmajor = colmajor_to_rowmajor(src2_data, dst_shape);

    NNGraph g("add_slice_bwd_pytorch");
    auto* src1 = g.tensor(src1_shape, "src1", DataType::FP32, true);
    auto* src2 = g.tensor(dst_shape, "src2", DataType::FP32, true);
    auto* out = add_slice(alpha, src1, beta, src2, "out", axis);

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
    std::vector<float> nntile_grad_src2_colmajor =
        runtime.get_output<float>(src2->grad()->name());
    std::vector<float> nntile_grad_src2 =
        colmajor_to_rowmajor(nntile_grad_src2_colmajor, dst_shape);

    std::vector<::int64_t> dst_shape_pt(dst_shape.begin(), dst_shape.end());
    std::vector<::int64_t> src1_shape_pt(src1_shape.begin(), src1_shape.end());
    auto src1_pt = torch::from_blob(src1_data.data(),
        src1_shape_pt,
        torch::TensorOptions().dtype(torch::kFloat32)).clone().set_requires_grad(true);
    auto src2_pt = torch::from_blob(src2_rowmajor.data(), dst_shape_pt,
        torch::TensorOptions().dtype(torch::kFloat32)).clone().set_requires_grad(true);

    auto src1_bc = src1_pt.unsqueeze(static_cast<int64_t>(axis))
                       .expand(dst_shape_pt);
    auto out_pt = alpha * src1_bc + beta * src2_pt;

    auto grad_output = torch::full(dst_shape_pt,
        static_cast<float>(grad_fill_val),
        torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false));
    out_pt.backward(grad_output);

    compare_float_vectors(nntile_grad_src1, src1_pt.grad());
    compare_float_vectors(nntile_grad_src2, src2_pt.grad());
}

#endif // NNTILE_HAVE_TORCH
