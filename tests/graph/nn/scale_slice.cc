/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/nn/scale_slice.cc
 * Test NNGraph scale_slice autograd operation.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#ifdef NNTILE_HAVE_TORCH
#   include "pytorch_helper.hh"
#   include "pytorch_tile_helpers.hh"
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
    "NNGraph scale_slice structure", "[graph][nn_graph]")
{
    const auto [alpha, axis] = GENERATE(
        std::tuple{Scalar(1.0), Index(0)},
        std::tuple{Scalar(2.0), Index(1)},
        std::tuple{Scalar(0.5), Index(0)},
        std::tuple{Scalar(-1.0), Index(1)});

    std::vector<Index> src_shape = slice_shape({dim_2, dim_4}, axis);
    Index axis_size = (axis == 0) ? dim_2 : dim_4;

    NNGraph g("scale_slice_structure");
    auto* src = g.tensor(src_shape, "src", DataType::FP32);
    auto* out = scale_slice(alpha, src, "out", axis, axis_size);

    REQUIRE(out != nullptr);
    REQUIRE(out->has_producer());
    REQUIRE(out->shape() == (std::vector<Index>{dim_2, dim_4}));
    REQUIRE(g.num_ops() == 1);
    REQUIRE(g.tensor_graph().ops()[0]->op_name() == "SCALE_SLICE");
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph scale_slice backward", "[graph][nn_graph]")
{
    const auto [alpha, axis, grad_fill_val] = GENERATE(
        std::tuple{Scalar(2.0), Index(1), Scalar(1.0)},
        std::tuple{Scalar(0.5), Index(0), Scalar(-1.0)},
        std::tuple{Scalar(1.0), Index(0), Scalar(0.5)},
        std::tuple{Scalar(-1.0), Index(1), Scalar(2.0)});

    std::vector<Index> src_shape = slice_shape({dim_2, dim_4}, axis);
    Index axis_size = (axis == 0) ? dim_2 : dim_4;

    NNGraph g("scale_slice_backward");
    auto* src = g.tensor(src_shape, "src", DataType::FP32);
    auto* out = scale_slice(alpha, src, "out", axis, axis_size);

    auto [out_grad, _] = g.get_or_create_grad(out, "out_grad");
    gt::fill(grad_fill_val, out_grad->data());
    out->backward();

    REQUIRE(src->has_grad());
    REQUIRE(src->grad()->shape() == src_shape);
}

#ifdef NNTILE_HAVE_TORCH

using nntile::test::compare_float_vectors;
using nntile::test::colmajor_to_rowmajor;
using nntile::test::nn_pytorch_tile_heterogeneous_1d_len6;
using nntile::test::nn_pytorch_tile_heterogeneous_1d_len7;
using nntile::test::nn_pytorch_tile_heterogeneous_rank2_6x7;
using nntile::test::pytorch_tolerance;

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph scale_slice forward matches PyTorch", "[graph][nn_graph][pytorch]")
{
    const auto [alpha, axis] = GENERATE(
        std::tuple{Scalar(1.0), Index(0)},
        std::tuple{Scalar(2.0), Index(1)},
        std::tuple{Scalar(0.5), Index(0)},
        std::tuple{Scalar(-1.0), Index(1)});

    constexpr Index dim_m = 6;
    constexpr Index dim_n = 7;
    std::vector<Index> dst_shape = {dim_m, dim_n};
    std::vector<Index> src_shape = slice_shape(dst_shape, axis);
    Index axis_size = (axis == 0) ? dim_m : dim_n;
    Index src_nelems = 1;
    for(Index d : src_shape)
        src_nelems *= d;
    const Index dst_nelems = dim_m * dim_n;

    std::vector<float> src_data(static_cast<size_t>(src_nelems));
    for(Index i = 0; i < src_nelems; ++i)
        src_data[i] = 0.1f * static_cast<float>(i + 1);

    NNGraph g("scale_slice_pytorch");
    auto* src = g.tensor(src_shape, "src", DataType::FP32, true);
    auto* out = scale_slice(alpha, src, "out", axis, axis_size);

    nn_pytorch_tile_heterogeneous_rank2_6x7(out);
    if(axis == 0)
        nn_pytorch_tile_heterogeneous_1d_len7(src);
    else
        nn_pytorch_tile_heterogeneous_1d_len6(src);

    src->mark_input(true);
    out->mark_output(true);

    TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());
    TileGraph::Runtime runtime(tile_graph);
    runtime.compile();
    runtime.bind_data("src", src_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> nntile_out_colmajor = runtime.get_output<float>("out");
    std::vector<float> nntile_out =
        colmajor_to_rowmajor(nntile_out_colmajor, dst_shape);

    std::vector<::int64_t> src_shape_pt(src_shape.begin(), src_shape.end());
    auto src_pt = torch::from_blob(src_data.data(), src_shape_pt,
        torch::TensorOptions().dtype(torch::kFloat32)).clone().set_requires_grad(false);
    auto out_pt = (alpha * src_pt.unsqueeze(static_cast<std::int64_t>(axis))
                       .expand({dim_m, dim_n})).contiguous();

    std::vector<float> pytorch_out(out_pt.data_ptr<float>(),
                                   out_pt.data_ptr<float>() + dst_nelems);

    REQUIRE(nntile_out.size() == pytorch_out.size());
    for(size_t i = 0; i < nntile_out.size(); ++i)
        REQUIRE(std::abs(nntile_out[i] - pytorch_out[i]) < pytorch_tolerance);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph scale_slice backward matches PyTorch", "[graph][nn_graph][pytorch]")
{
    const auto [alpha, axis, grad_fill_val] = GENERATE(
        std::tuple{Scalar(2.0), Index(1), Scalar(1.0)},
        std::tuple{Scalar(0.5), Index(0), Scalar(-1.0)},
        std::tuple{Scalar(1.0), Index(0), Scalar(0.5)},
        std::tuple{Scalar(-1.0), Index(1), Scalar(2.0)});

    constexpr Index dim_m = 6;
    constexpr Index dim_n = 7;
    std::vector<Index> dst_shape = {dim_m, dim_n};
    std::vector<Index> src_shape = slice_shape(dst_shape, axis);
    Index axis_size = (axis == 0) ? dim_m : dim_n;
    Index src_nelems = 1;
    for(Index d : src_shape)
        src_nelems *= d;

    std::vector<float> src_data(static_cast<size_t>(src_nelems));
    for(Index i = 0; i < src_nelems; ++i)
        src_data[i] = 0.1f * static_cast<float>(i);

    NNGraph g("scale_slice_bwd_pytorch");
    auto* src = g.tensor(src_shape, "src", DataType::FP32, true);
    auto* out = scale_slice(alpha, src, "out", axis, axis_size);

    nn_pytorch_tile_heterogeneous_rank2_6x7(out);
    if(axis == 0)
        nn_pytorch_tile_heterogeneous_1d_len7(src);
    else
        nn_pytorch_tile_heterogeneous_1d_len6(src);

    src->mark_input(true);

    auto [out_grad, _] = g.get_or_create_grad(out, "out_grad");
    gt::fill(grad_fill_val, out_grad->data());
    out->backward();

    src->grad()->mark_output(true);

    TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());
    TileGraph::Runtime runtime(tile_graph);
    runtime.compile();
    runtime.bind_data("src", src_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> nntile_grad_src =
        runtime.get_output<float>(src->grad()->name());

    std::vector<::int64_t> src_shape_pt(src_shape.begin(), src_shape.end());
    auto src_pt = torch::from_blob(src_data.data(), src_shape_pt,
        torch::TensorOptions().dtype(torch::kFloat32)).clone().set_requires_grad(true);
    auto out_pt = alpha * src_pt.unsqueeze(static_cast<std::int64_t>(axis))
                      .expand({dim_m, dim_n});

    auto grad_output = torch::full({dim_m, dim_n}, static_cast<float>(grad_fill_val),
        torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false));
    out_pt.backward(grad_output);

    compare_float_vectors(nntile_grad_src, src_pt.grad());
}

#endif // NNTILE_HAVE_TORCH
