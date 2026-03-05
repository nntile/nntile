/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/nn/multiply_slice.cc
 * Test NNGraph multiply_slice autograd operation.
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
    for(Index i = 0; i < static_cast<Index>(dst_shape.size()); ++i)
        if(i != axis)
            out.push_back(dst_shape[i]);
    return out;
}

constexpr Index dim_2 = 2;
constexpr Index dim_4 = 4;

} // anonymous namespace

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph multiply_slice structure", "[graph][nn_graph]")
{
    const auto [alpha, axis] = GENERATE(
        std::tuple{Scalar(1.0), Index(0)},
        std::tuple{Scalar(1.0), Index(1)},
        std::tuple{Scalar(2.0), Index(0)},
        std::tuple{Scalar(0.5), Index(1)});

    std::vector<Index> slice_sh = slice_shape({dim_2, dim_4}, axis);
    std::vector<Index> tensor_shape = {dim_2, dim_4};

    NNGraph g("multiply_slice_structure");
    auto* slice_node = g.tensor(slice_sh, "slice", DataType::FP32);
    auto* tensor_node = g.tensor(tensor_shape, "tensor", DataType::FP32);
    fill(1.0, tensor_node);
    auto* out = multiply_slice(alpha, slice_node, tensor_node, "out", axis);

    REQUIRE(out != nullptr);
    REQUIRE(out->has_producer());
    REQUIRE(out->shape() == (std::vector<Index>{dim_2, dim_4}));
    REQUIRE(g.num_ops() == 1);
    auto has_multiply_slice = [&g]() {
        for(const auto& op : g.tensor_graph().ops())
            if(op->op_name() == "MULTIPLY_SLICE")
                return true;
        return false;
    };
    REQUIRE(has_multiply_slice());
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph multiply_slice backward", "[graph][nn_graph]")
{
    const auto [alpha, axis, grad_fill_val] = GENERATE(
        std::tuple{Scalar(1.0), Index(1), Scalar(1.0)},
        std::tuple{Scalar(1.0), Index(0), Scalar(-1.0)},
        std::tuple{Scalar(2.0), Index(0), Scalar(0.5)},
        std::tuple{Scalar(0.5), Index(1), Scalar(2.0)});

    std::vector<Index> slice_sh = slice_shape({dim_2, dim_4}, axis);
    std::vector<Index> tensor_shape = {dim_2, dim_4};

    NNGraph g("multiply_slice_backward");
    auto* slice_node = g.tensor(slice_sh, "slice", DataType::FP32);
    auto* tensor_node = g.tensor(tensor_shape, "tensor", DataType::FP32);
    fill(1.0, tensor_node);
    auto* out = multiply_slice(alpha, slice_node, tensor_node, "out", axis);

    auto [out_grad, _] = g.get_or_create_grad(out, "out_grad");
    gt::fill(grad_fill_val, out_grad->data());
    out->backward();

    REQUIRE(slice_node->has_grad());
    REQUIRE(slice_node->grad()->shape() == slice_sh);
}

#ifdef NNTILE_HAVE_TORCH

using nntile::test::compare_float_vectors;
using nntile::test::colmajor_to_rowmajor;
using nntile::test::pytorch_tolerance;

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph multiply_slice forward matches PyTorch", "[graph][nn_graph][pytorch]")
{
    const auto [alpha, axis] = GENERATE(
        std::tuple{Scalar(1.0), Index(0)},
        std::tuple{Scalar(1.0), Index(1)},
        std::tuple{Scalar(2.0), Index(0)},
        std::tuple{Scalar(0.5), Index(1)});

    std::vector<Index> slice_sh = slice_shape({dim_2, dim_4}, axis);
    Index slice_nelems = 1;
    for(Index d : slice_sh)
        slice_nelems *= d;
    const Index dst_nelems = dim_2 * dim_4;

    std::vector<float> slice_data(static_cast<size_t>(slice_nelems));
    for(Index i = 0; i < slice_nelems; ++i)
        slice_data[i] = 0.1f * static_cast<float>(i + 1);

    NNGraph g("multiply_slice_pytorch");
    auto* slice_node = g.tensor(slice_sh, "slice", DataType::FP32, true);
    auto* tensor_node = g.tensor({dim_2, dim_4}, "tensor", DataType::FP32);
    fill(1.0, tensor_node);
    auto* out = multiply_slice(alpha, slice_node, tensor_node, "out", axis);

    slice_node->mark_input(true);
    out->mark_output(true);

    TensorGraph::Runtime runtime(g.tensor_graph());
    runtime.compile();
    runtime.bind_data("slice", slice_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> nntile_out_colmajor = runtime.get_output<float>("out");
    std::vector<float> nntile_out =
        colmajor_to_rowmajor(nntile_out_colmajor, {dim_2, dim_4});

    std::vector<::int64_t> slice_shape_pt(slice_sh.begin(), slice_sh.end());
    auto slice_pt = torch::from_blob(slice_data.data(), slice_shape_pt,
        torch::TensorOptions().dtype(torch::kFloat32)).clone().set_requires_grad(false);
    auto out_pt = (alpha * slice_pt.unsqueeze(static_cast<std::int64_t>(axis))
                      .expand({dim_2, dim_4})).contiguous();

    std::vector<float> pytorch_out(out_pt.data_ptr<float>(),
                                   out_pt.data_ptr<float>() + dst_nelems);

    REQUIRE(nntile_out.size() == pytorch_out.size());
    for(size_t i = 0; i < nntile_out.size(); ++i)
        REQUIRE(std::abs(nntile_out[i] - pytorch_out[i]) < pytorch_tolerance);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph multiply_slice backward matches PyTorch", "[graph][nn_graph][pytorch]")
{
    const auto [alpha, axis, grad_fill_val] = GENERATE(
        std::tuple{Scalar(1.0), Index(1), Scalar(1.0)},
        std::tuple{Scalar(1.0), Index(0), Scalar(-1.0)},
        std::tuple{Scalar(2.0), Index(0), Scalar(0.5)},
        std::tuple{Scalar(0.5), Index(1), Scalar(2.0)});

    std::vector<Index> slice_sh = slice_shape({dim_2, dim_4}, axis);
    Index slice_nelems = 1;
    for(Index d : slice_sh)
        slice_nelems *= d;

    std::vector<float> slice_data(static_cast<size_t>(slice_nelems));
    for(Index i = 0; i < slice_nelems; ++i)
        slice_data[i] = 0.1f * static_cast<float>(i);

    NNGraph g("multiply_slice_bwd_pytorch");
    auto* slice_node = g.tensor(slice_sh, "slice", DataType::FP32, true);
    auto* tensor_node = g.tensor({dim_2, dim_4}, "tensor", DataType::FP32);
    fill(1.0, tensor_node);
    auto* out = multiply_slice(alpha, slice_node, tensor_node, "out", axis);

    slice_node->mark_input(true);

    auto [out_grad, _] = g.get_or_create_grad(out, "out_grad");
    gt::fill(grad_fill_val, out_grad->data());
    out->backward();

    slice_node->grad()->mark_output(true);

    TensorGraph::Runtime runtime(g.tensor_graph());
    runtime.compile();
    runtime.bind_data("slice", slice_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> nntile_grad_slice =
        runtime.get_output<float>(slice_node->grad()->name());

    std::vector<::int64_t> slice_shape_pt(slice_sh.begin(), slice_sh.end());
    auto slice_pt = torch::from_blob(slice_data.data(), slice_shape_pt,
        torch::TensorOptions().dtype(torch::kFloat32)).clone().set_requires_grad(true);
    auto out_pt = alpha * slice_pt.unsqueeze(static_cast<std::int64_t>(axis)).expand({dim_2, dim_4});

    auto grad_output = torch::full({dim_2, dim_4}, static_cast<float>(grad_fill_val),
        torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false));
    out_pt.backward(grad_output);

    compare_float_vectors(nntile_grad_slice, slice_pt.grad());
}

#endif // NNTILE_HAVE_TORCH
