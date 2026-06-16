/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/nn_graph/gemm.cc
 * Test NNGraph gemm autograd operation.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <numeric>

#ifdef NNTILE_HAVE_TORCH
#include "gemm_test_shapes.hh"
#include "pytorch_helper.hh"
#include "pytorch_gemm_helper.hh"
#endif

#include "context_fixture.hh"
#include "nntile/graph.hh"

using namespace nntile;
using namespace nntile;
namespace gt = nntile::tensor;

namespace
{

constexpr Scalar gemm_alpha_one = 1.0;
constexpr bool trans_a_default = false;
constexpr bool trans_b_default = false;
constexpr Index ndim_one = 1;
constexpr Index ndim_two = 2;
constexpr Index batch_ndim_none = 0;
constexpr Index batch_ndim_one = 1;

} // anonymous namespace

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph gemm structure",
    "[graph][nn_graph]")
{
    const auto [M, K, N] = GENERATE(std::tuple{Index(2), Index(3), Index(4)},
        std::tuple{Index(3), Index(4), Index(3)});

    NNGraph g("gemm_structure");
    auto *a = g.tensor({K, M}, DataType::FP32)->set_name("a");
    auto *b = g.tensor({K, N}, DataType::FP32)->set_name("b");
    auto *c = gemm(a,
        b,
        gemm_alpha_one,
        trans_a_default,
        trans_b_default,
        ndim_one,
        batch_ndim_none);

    REQUIRE(c != nullptr);
    REQUIRE(c->has_producer());
    REQUIRE(c->shape() == (std::vector<Index>{N, M}));
    REQUIRE(g.num_ops() == 1);
    REQUIRE(g.tensor_graph().ops()[0]->op_name() == "GEMM");
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph gemm structure multi-dimensional",
    "[graph][nn_graph]")
{
    SECTION("ndim=2, batch_ndim=0: 4D contraction")
    {
        const Index M1 = 2, M2 = 3, K1 = 4, K2 = 2, N1 = 3, N2 = 5;
        NNGraph g("gemm_4d");
        auto *a = g.tensor({K1, K2, M2, M1}, DataType::FP32)->set_name("a");
        auto *b = g.tensor({K1, K2, N1, N2}, DataType::FP32)->set_name("b");
        auto *c = gemm(a,
            b,
            gemm_alpha_one,
            trans_a_default,
            trans_b_default,
            ndim_two,
            batch_ndim_none);
        REQUIRE(c != nullptr);
        REQUIRE(c->has_producer());
        REQUIRE(c->shape() == (std::vector<Index>{N1, N2, M2, M1}));
        REQUIRE(g.num_ops() == 1);
    }
    SECTION("ndim=1, batch_ndim=1: batched 2D matrices")
    {
        const Index B = 4, M = 2, K = 3, N = 5;
        NNGraph g("gemm_batched");
        auto *a = g.tensor({B, K, M}, DataType::FP32)->set_name("a");
        auto *b = g.tensor({B, K, N}, DataType::FP32)->set_name("b");
        auto *c = gemm(a,
            b,
            gemm_alpha_one,
            trans_a_default,
            trans_b_default,
            ndim_one,
            batch_ndim_one);
        REQUIRE(c != nullptr);
        REQUIRE(c->has_producer());
        REQUIRE(c->shape() == (std::vector<Index>{B, N, M}));
        REQUIRE(g.num_ops() == 1);
    }
    SECTION("ndim=2, batch_ndim=0: a.ndim() != b.ndim() (3D @ 4D)")
    {
        const Index M1 = 2, K1 = 3, K2 = 4, N1 = 5, N2 = 6;
        NNGraph g("gemm_3d_4d");
        auto *a = g.tensor({K1, K2, M1}, DataType::FP32)->set_name("a");
        auto *b = g.tensor({K1, K2, N1, N2}, DataType::FP32)->set_name("b");
        auto *c = gemm(a,
            b,
            gemm_alpha_one,
            trans_a_default,
            trans_b_default,
            ndim_two,
            batch_ndim_none);
        REQUIRE(c != nullptr);
        REQUIRE(c->has_producer());
        REQUIRE(a->ndim() == 3);
        REQUIRE(b->ndim() == 4);
        REQUIRE(c->shape() == (std::vector<Index>{N1, N2, M1}));
        REQUIRE(g.num_ops() == 1);
    }
}

TEST_CASE_METHOD(
    nntile::test::ContextFixture, "NNGraph gemm backward", "[graph][nn_graph]")
{
    const auto [M, K, N, grad_fill_val] =
        GENERATE(std::tuple{Index(2), Index(3), Index(4), Scalar(1.0)},
            std::tuple{Index(3), Index(4), Index(3), Scalar(-1.0)});

    NNGraph g("gemm_backward");
    auto *a = g.tensor({K, M}, DataType::FP32)->set_name("a");
    auto *b = g.tensor({K, N}, DataType::FP32)->set_name("b");
    auto *c = gemm(a,
        b,
        gemm_alpha_one,
        trans_a_default,
        trans_b_default,
        ndim_one,
        batch_ndim_none);

    auto [c_grad, _] = g.get_or_create_grad(c, "c_grad");
    gt::fill(grad_fill_val, c_grad->data());
    c->backward();

    REQUIRE(a->has_grad());
    REQUIRE(b->has_grad());
    REQUIRE(a->grad()->shape() == (std::vector<Index>{K, M}));
    REQUIRE(b->grad()->shape() == (std::vector<Index>{K, N}));
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph gemm backward multi-dimensional",
    "[graph][nn_graph]")
{
    SECTION("ndim=2, batch_ndim=0")
    {
        const Index M1 = 1, M2 = 2, K1 = 3, K2 = 4, N1 = 5, N2 = 6;
        NNGraph g("gemm_bwd_4d");
        auto *a = g.tensor({K1, K2, M2, M1}, DataType::FP32)->set_name("a");
        auto *b = g.tensor({K1, K2, N1, N2}, DataType::FP32)->set_name("b");
        auto *c = gemm(a,
            b,
            gemm_alpha_one,
            trans_a_default,
            trans_b_default,
            ndim_two,
            batch_ndim_none);
        auto [c_grad, _] = g.get_or_create_grad(c, "c_grad");
        gt::fill(Scalar(1.0), c_grad->data());
        c->backward();
        REQUIRE(a->has_grad());
        REQUIRE(b->has_grad());
        REQUIRE(a->grad()->shape() == (std::vector<Index>{K1, K2, M2, M1}));
        REQUIRE(b->grad()->shape() == (std::vector<Index>{K1, K2, N1, N2}));
    }
    SECTION("ndim=1, batch_ndim=1")
    {
        const Index B = 3, M = 2, K = 4, N = 3;
        NNGraph g("gemm_bwd_batched");
        auto *a = g.tensor({B, K, M}, DataType::FP32)->set_name("a");
        auto *b = g.tensor({B, K, N}, DataType::FP32)->set_name("b");
        auto *c = gemm(a,
            b,
            gemm_alpha_one,
            trans_a_default,
            trans_b_default,
            ndim_one,
            batch_ndim_one);
        auto [c_grad, _] = g.get_or_create_grad(c, "c_grad");
        gt::fill(Scalar(-1.0), c_grad->data());
        c->backward();
        REQUIRE(a->has_grad());
        REQUIRE(b->has_grad());
        REQUIRE(a->grad()->shape() == (std::vector<Index>{B, K, M}));
        REQUIRE(b->grad()->shape() == (std::vector<Index>{B, K, N}));
    }
    SECTION("ndim=2, batch_ndim=0: a.ndim() != b.ndim() (3D @ 4D)")
    {
        const Index M1 = 2, K1 = 3, K2 = 4, N1 = 5, N2 = 6;
        NNGraph g("gemm_bwd_3d_4d");
        auto *a = g.tensor({K1, K2, M1}, DataType::FP32)->set_name("a");
        auto *b = g.tensor({K1, K2, N1, N2}, DataType::FP32)->set_name("b");
        auto *c = gemm(a,
            b,
            gemm_alpha_one,
            trans_a_default,
            trans_b_default,
            ndim_two,
            batch_ndim_none);
        REQUIRE(c != nullptr);
        REQUIRE(c->shape() == (std::vector<Index>{N1, N2, M1}));
    }
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph gemm forward and backward",
    "[graph][nn_graph]")
{
    const auto [M, K, N, gemm_alpha, grad_fill_val] = GENERATE(
        std::tuple{Index(2), Index(3), Index(4), Scalar(1.0), Scalar(1.0)},
        std::tuple{Index(3), Index(4), Index(3), Scalar(0.5), Scalar(1.0)},
        std::tuple{Index(4), Index(5), Index(6), Scalar(2.0), Scalar(-1.0)});

    NNGraph g("gemm");
    auto *a = g.tensor({K, M}, DataType::FP32, true)->set_name("a");
    auto *b = g.tensor({K, N}, DataType::FP32, true)->set_name("b");
    auto *c = gemm(a,
        b,
        gemm_alpha,
        trans_a_default,
        trans_b_default,
        ndim_one,
        batch_ndim_none);

    REQUIRE(c != nullptr);
    REQUIRE(c->has_producer());
    REQUIRE(c->shape() == (std::vector<Index>{N, M}));

    auto [c_grad, _] = g.get_or_create_grad(c, "c_grad");
    gt::fill(grad_fill_val, c_grad->data());
    c->backward();

    REQUIRE(a->has_grad());
    REQUIRE(b->has_grad());
    REQUIRE(a->grad()->shape() == (std::vector<Index>{K, M}));
    REQUIRE(b->grad()->shape() == (std::vector<Index>{K, N}));
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph gemm forward and backward multi-dimensional",
    "[graph][nn_graph]")
{
    const auto [a_shape,
        b_shape,
        expected_c_shape,
        ndim,
        batch_ndim,
        alpha,
        grad_val] = GENERATE(std::tuple{std::vector<Index>{2, 4, 3, 2},
                                 std::vector<Index>{2, 4, 5, 3},
                                 std::vector<Index>{5, 3, 3, 2},
                                 ndim_two,
                                 batch_ndim_none,
                                 Scalar(1.0),
                                 Scalar(1.0)},
        std::tuple{std::vector<Index>{3, 4, 2},
            std::vector<Index>{3, 4, 3},
            std::vector<Index>{3, 3, 2},
            ndim_one,
            batch_ndim_one,
            Scalar(0.5),
            Scalar(-1.0)},
        std::tuple{std::vector<Index>{2, 4, 3, 2},
            std::vector<Index>{2, 4, 5, 6},
            std::vector<Index>{5, 6, 3, 2},
            ndim_two,
            batch_ndim_none,
            Scalar(1.0),
            Scalar(1.0)});

    NNGraph g("gemm_md");
    auto *a = g.tensor(a_shape, DataType::FP32, true)->set_name("a");
    auto *b = g.tensor(b_shape, DataType::FP32, true)->set_name("b");
    auto *c =
        gemm(a, b, alpha, trans_a_default, trans_b_default, ndim, batch_ndim);

    REQUIRE(c != nullptr);
    REQUIRE(c->shape() == expected_c_shape);

    auto [c_grad, _] = g.get_or_create_grad(c, "c_grad");
    gt::fill(grad_val, c_grad->data());
    c->backward();

    REQUIRE(a->has_grad());
    REQUIRE(b->has_grad());
    REQUIRE(a->grad()->shape() == a_shape);
    REQUIRE(b->grad()->shape() == b_shape);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph gemm backward transposed A",
    "[graph][nn_graph]")
{
    constexpr Scalar grad_fill_val = 1.0;

    SECTION("trans_b=false uses grad_C transpose for grad_A")
    {
        constexpr bool trans_a = true;
        constexpr bool trans_b = false;

        NNGraph g("gemm_transposed_a");
        auto *a = g.tensor({2, 4}, DataType::FP32, true)->set_name("a");
        auto *b = g.tensor({4, 3}, DataType::FP32, true)->set_name("b");
        auto *c = gemm(
            a, b, gemm_alpha_one, trans_a, trans_b, ndim_one, batch_ndim_none);

        REQUIRE(c != nullptr);
        REQUIRE(c->shape() == (std::vector<Index>{3, 2}));

        auto [c_grad, _] = g.get_or_create_grad(c, "c_grad");
        gt::fill(grad_fill_val, c_grad->data());

        REQUIRE_NOTHROW(c->backward());
        REQUIRE(a->has_grad());
        REQUIRE(b->has_grad());
        REQUIRE(a->grad()->shape() == (std::vector<Index>{2, 4}));
        REQUIRE(b->grad()->shape() == (std::vector<Index>{4, 3}));
    }

    SECTION("trans_b=true uses grad_C transpose for grad_A")
    {
        constexpr bool trans_a = true;
        constexpr bool trans_b = true;

        NNGraph g("gemm_transposed_a_transposed_b");
        auto *a = g.tensor({2, 4}, DataType::FP32, true)->set_name("a");
        auto *b = g.tensor({3, 4}, DataType::FP32, true)->set_name("b");
        auto *c = gemm(
            a, b, gemm_alpha_one, trans_a, trans_b, ndim_one, batch_ndim_none);

        REQUIRE(c != nullptr);
        REQUIRE(c->shape() == (std::vector<Index>{3, 2}));

        auto [c_grad, _] = g.get_or_create_grad(c, "c_grad");
        gt::fill(grad_fill_val, c_grad->data());

        REQUIRE_NOTHROW(c->backward());
        REQUIRE(a->has_grad());
        REQUIRE(b->has_grad());
        REQUIRE(a->grad()->shape() == (std::vector<Index>{2, 4}));
        REQUIRE(b->grad()->shape() == (std::vector<Index>{3, 4}));
    }
}

#ifdef NNTILE_HAVE_TORCH

#include "gemm_test_shapes.hh"
#include "pytorch_helper.hh"
#include "pytorch_gemm_helper.hh"

using nntile::test::require_relative_frobenius_error;
using nntile::test::gemm_test_shapes;
using nntile::test::pytorch_gemm_reference;

constexpr float pytorch_gemm_tolerance = 1e-6f;

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph gemm matches PyTorch for all flag combos",
    "[graph][nn_graph][pytorch]")
{
    const auto [trans_a, trans_b, ndim, batch_ndim, gemm_alpha] = GENERATE(
        std::tuple{false, false, Index(1), Index(0), Scalar(1.0)},
        std::tuple{false, true, Index(1), Index(0), Scalar(1.0)},
        std::tuple{true, false, Index(1), Index(0), Scalar(1.0)},
        std::tuple{true, true, Index(1), Index(0), Scalar(1.0)},
        std::tuple{false, false, Index(2), Index(0), Scalar(0.5)},
        std::tuple{false, true, Index(2), Index(0), Scalar(0.5)},
        std::tuple{true, false, Index(2), Index(0), Scalar(0.5)},
        std::tuple{true, true, Index(2), Index(0), Scalar(0.5)},
        std::tuple{false, false, Index(1), Index(1), Scalar(1.0)},
        std::tuple{false, true, Index(1), Index(1), Scalar(1.0)},
        std::tuple{true, false, Index(1), Index(1), Scalar(1.0)},
        std::tuple{true, true, Index(1), Index(1), Scalar(1.0)},
        std::tuple{false, false, Index(1), Index(2), Scalar(1.0)},
        std::tuple{false, true, Index(1), Index(2), Scalar(1.0)},
        std::tuple{true, false, Index(1), Index(2), Scalar(1.0)},
        std::tuple{true, true, Index(1), Index(2), Scalar(1.0)});

    const auto [a_shape, b_shape] =
        gemm_test_shapes(trans_a, trans_b, ndim, batch_ndim);
    const Index a_nelems = std::accumulate(a_shape.begin(),
        a_shape.end(),
        Index{1},
        std::multiplies<Index>{});
    const Index b_nelems = std::accumulate(b_shape.begin(),
        b_shape.end(),
        Index{1},
        std::multiplies<Index>{});

    std::vector<float> a_data(static_cast<size_t>(a_nelems));
    std::vector<float> b_data(static_cast<size_t>(b_nelems));
    for (Index i = 0; i < a_nelems; ++i)
        a_data[static_cast<size_t>(i)] = 0.1f * static_cast<float>(i + 1);
    for (Index i = 0; i < b_nelems; ++i)
        b_data[static_cast<size_t>(i)] = 0.15f * static_cast<float>(i + 2);

    NNGraph g("gemm_pytorch_all");
    auto *a = g.tensor(a_shape, DataType::FP32, true)->set_name("a");
    auto *b = g.tensor(b_shape, DataType::FP32, true)->set_name("b");
    auto *c = gemm(a, b, gemm_alpha, trans_a, trans_b, ndim, batch_ndim);

    a->mark_input(true);
    b->mark_input(true);
    c->mark_output(true);

    TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());
    Runtime runtime(tile_graph);
    runtime.compile();
    runtime.bind_data(a, a_data);
    runtime.bind_data(b, b_data);
    runtime.execute();
    runtime.wait();

    std::vector<::int64_t> a_sizes;
    std::vector<::int64_t> b_sizes;
    a_sizes.reserve(a_shape.size());
    b_sizes.reserve(b_shape.size());
    for (Index dim : a_shape)
        a_sizes.push_back(static_cast<::int64_t>(dim));
    for (Index dim : b_shape)
        b_sizes.push_back(static_cast<::int64_t>(dim));
    auto a_pt = torch::from_blob(a_data.data(),
        a_sizes,
        torch::TensorOptions().dtype(torch::kFloat32))
                    .clone();
    auto b_pt = torch::from_blob(b_data.data(),
        b_sizes,
        torch::TensorOptions().dtype(torch::kFloat32))
                    .clone();
    auto ref_pt = pytorch_gemm_reference(
        a_pt, b_pt, trans_a, trans_b, ndim, batch_ndim, gemm_alpha);

    require_relative_frobenius_error(
        runtime.get_output<float>(c), ref_pt, pytorch_gemm_tolerance);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph gemm backward matches PyTorch for all flag combos",
    "[graph][nn_graph][pytorch]")
{
    const auto [trans_a, trans_b, ndim, batch_ndim, gemm_alpha, grad_fill_val] =
        GENERATE(
            std::tuple{
                false, false, Index(1), Index(0), Scalar(1.0), Scalar(1.0)},
            std::tuple{
                false, true, Index(1), Index(0), Scalar(1.0), Scalar(1.0)},
            std::tuple{
                true, false, Index(1), Index(0), Scalar(1.0), Scalar(1.0)},
            std::tuple{
                true, true, Index(1), Index(0), Scalar(1.0), Scalar(1.0)},
            std::tuple{
                false, false, Index(2), Index(0), Scalar(0.5), Scalar(-1.0)},
            std::tuple{
                false, true, Index(2), Index(0), Scalar(0.5), Scalar(-1.0)},
            std::tuple{
                true, false, Index(2), Index(0), Scalar(0.5), Scalar(-1.0)},
            std::tuple{
                true, true, Index(2), Index(0), Scalar(0.5), Scalar(-1.0)},
            std::tuple{
                false, false, Index(1), Index(1), Scalar(1.0), Scalar(1.0)},
            std::tuple{
                false, true, Index(1), Index(1), Scalar(1.0), Scalar(1.0)},
            std::tuple{
                true, false, Index(1), Index(1), Scalar(1.0), Scalar(1.0)},
            std::tuple{
                true, true, Index(1), Index(1), Scalar(1.0), Scalar(1.0)},
            std::tuple{
                false, false, Index(1), Index(2), Scalar(1.0), Scalar(1.0)},
            std::tuple{
                false, true, Index(1), Index(2), Scalar(1.0), Scalar(1.0)},
            std::tuple{
                true, false, Index(1), Index(2), Scalar(1.0), Scalar(1.0)},
            std::tuple{
                true, true, Index(1), Index(2), Scalar(1.0), Scalar(1.0)});

    const auto [a_shape, b_shape] =
        gemm_test_shapes(trans_a, trans_b, ndim, batch_ndim);
    const Index a_nelems = std::accumulate(a_shape.begin(),
        a_shape.end(),
        Index{1},
        std::multiplies<Index>{});
    const Index b_nelems = std::accumulate(b_shape.begin(),
        b_shape.end(),
        Index{1},
        std::multiplies<Index>{});

    std::vector<float> a_data(static_cast<size_t>(a_nelems));
    std::vector<float> b_data(static_cast<size_t>(b_nelems));
    for (Index i = 0; i < a_nelems; ++i)
        a_data[static_cast<size_t>(i)] = 0.1f * static_cast<float>(i + 1);
    for (Index i = 0; i < b_nelems; ++i)
        b_data[static_cast<size_t>(i)] = 0.15f * static_cast<float>(i + 2);

    NNGraph g("gemm_bwd_pytorch_all");
    auto *a = g.tensor(a_shape, DataType::FP32, true)->set_name("a");
    auto *b = g.tensor(b_shape, DataType::FP32, true)->set_name("b");
    auto *c = gemm(a, b, gemm_alpha, trans_a, trans_b, ndim, batch_ndim);

    a->mark_input(true);
    b->mark_input(true);

    auto [c_grad, _] = g.get_or_create_grad(c, "c_grad");
    gt::fill(grad_fill_val, c_grad->data());
    c->backward();

    a->grad()->mark_output(true);
    b->grad()->mark_output(true);

    TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());
    Runtime runtime(tile_graph);
    runtime.compile();
    runtime.bind_data(a, a_data);
    runtime.bind_data(b, b_data);
    runtime.execute();
    runtime.wait();

    std::vector<::int64_t> a_sizes;
    std::vector<::int64_t> b_sizes;
    a_sizes.reserve(a_shape.size());
    b_sizes.reserve(b_shape.size());
    for (Index dim : a_shape)
        a_sizes.push_back(static_cast<::int64_t>(dim));
    for (Index dim : b_shape)
        b_sizes.push_back(static_cast<::int64_t>(dim));
    auto a_pt = torch::from_blob(a_data.data(),
        a_sizes,
        torch::TensorOptions().dtype(torch::kFloat32))
                    .clone()
                    .set_requires_grad(true);
    auto b_pt = torch::from_blob(b_data.data(),
        b_sizes,
        torch::TensorOptions().dtype(torch::kFloat32))
                    .clone()
                    .set_requires_grad(true);
    auto out_pt = pytorch_gemm_reference(
        a_pt, b_pt, trans_a, trans_b, ndim, batch_ndim, gemm_alpha);

    auto grad_output = torch::full(out_pt.sizes(),
        static_cast<float>(grad_fill_val),
        torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false));
    out_pt.backward(grad_output);

    require_relative_frobenius_error(runtime.get_output<float>(a->grad()),
        a_pt.grad(),
        pytorch_gemm_tolerance);
    require_relative_frobenius_error(runtime.get_output<float>(b->grad()),
        b_pt.grad(),
        pytorch_gemm_tolerance);
}


TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph gemm linear pattern matches PyTorch",
    "[graph][nn_graph][pytorch]")
{
    constexpr Index batch = 2;
    constexpr Index in_dim = 3;
    constexpr Index out_dim = 4;
    constexpr Scalar gemm_alpha = 1.0;
    constexpr Scalar grad_fill_val = 1.0;

    std::vector<float> weight_data(out_dim * in_dim);
    std::vector<float> input_data(batch * in_dim);
    for (Index i = 0; i < out_dim * in_dim; ++i)
        weight_data[static_cast<size_t>(i)] = 0.1f * static_cast<float>(i + 1);
    for (Index i = 0; i < batch * in_dim; ++i)
        input_data[static_cast<size_t>(i)] = 0.15f * static_cast<float>(i + 2);

    NNGraph g("gemm_linear");
    auto *weight =
        g.tensor({out_dim, in_dim}, DataType::FP32, true)->set_name("weight");
    auto *input =
        g.tensor({batch, in_dim}, DataType::FP32, true)->set_name("input");
    auto *output = gemm(
        weight, input, gemm_alpha, true, true, ndim_one, batch_ndim_none);

    auto [output_grad, _] = g.get_or_create_grad(output, "output_grad");
    gt::fill(grad_fill_val, output_grad->data());
    output->backward();

    weight->mark_input(true);
    input->mark_input(true);
    output->mark_output(true);
    weight->grad()->mark_output(true);
    input->grad()->mark_output(true);

    TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());
    Runtime runtime(tile_graph);
    runtime.compile();
    runtime.bind_data(weight, weight_data);
    runtime.bind_data(input, input_data);
    runtime.execute();
    runtime.wait();

    auto weight_pt = torch::from_blob(weight_data.data(),
        {out_dim, in_dim},
        torch::TensorOptions().dtype(torch::kFloat32))
                         .clone()
                         .set_requires_grad(true);
    auto input_pt = torch::from_blob(input_data.data(),
        {batch, in_dim},
        torch::TensorOptions().dtype(torch::kFloat32))
                        .clone()
                        .set_requires_grad(true);
    auto out_pt = pytorch_gemm_reference(weight_pt,
        input_pt,
        true,
        true,
        ndim_one,
        batch_ndim_none,
        gemm_alpha);

    auto grad_output = torch::full(out_pt.sizes(),
        static_cast<float>(grad_fill_val),
        torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false));
    out_pt.backward(grad_output);

    require_relative_frobenius_error(
        runtime.get_output<float>(output), out_pt, pytorch_gemm_tolerance);
    require_relative_frobenius_error(runtime.get_output<float>(weight->grad()),
        weight_pt.grad(),
        pytorch_gemm_tolerance);
    require_relative_frobenius_error(runtime.get_output<float>(input->grad()),
        input_pt.grad(),
        pytorch_gemm_tolerance);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph gemm attention Q pattern matches PyTorch",
    "[graph][nn_graph][pytorch]")
{
    constexpr Index batch = 2;
    constexpr Index seq = 8;
    constexpr Index hidden = 64;
    constexpr Index head_size = 16;
    constexpr Index n_heads = 4;
    constexpr Scalar gemm_alpha = 1.0;
    constexpr Scalar grad_fill_val = 1.0;

    const Index w_nelems = hidden * head_size * n_heads;
    const Index x_nelems = batch * seq * hidden;
    std::vector<float> w_data(static_cast<size_t>(w_nelems));
    std::vector<float> x_data(static_cast<size_t>(x_nelems));
    for (Index i = 0; i < w_nelems; ++i)
        w_data[static_cast<size_t>(i)] = 0.1f * static_cast<float>(i + 1);
    for (Index i = 0; i < x_nelems; ++i)
        x_data[static_cast<size_t>(i)] = 0.15f * static_cast<float>(i + 2);

    NNGraph g("gemm_attn_q");
    auto *w = g.tensor({hidden, head_size, n_heads}, DataType::FP32, true)
                  ->set_name("w");
    auto *x = g.tensor({batch, seq, hidden}, DataType::FP32, true)->set_name("x");
    auto *c = gemm(w, x, gemm_alpha, false, true, ndim_one, batch_ndim_none);

    auto [c_grad, _] = g.get_or_create_grad(c, "grad_c");
    gt::fill(grad_fill_val, c_grad->data());
    c->backward();

    w->mark_input(true);
    x->mark_input(true);
    c->mark_output(true);
    w->grad()->mark_output(true);
    x->grad()->mark_output(true);

    TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());
    Runtime runtime(tile_graph);
    runtime.compile();
    runtime.bind_data(w, w_data);
    runtime.bind_data(x, x_data);
    runtime.execute();
    runtime.wait();

    auto w_pt = torch::from_blob(w_data.data(),
        {hidden, head_size, n_heads},
        torch::TensorOptions().dtype(torch::kFloat32))
                    .clone()
                    .set_requires_grad(true);
    auto x_pt = torch::from_blob(x_data.data(),
        {batch, seq, hidden},
        torch::TensorOptions().dtype(torch::kFloat32))
                    .clone()
                    .set_requires_grad(true);
    auto out_pt = pytorch_gemm_reference(w_pt,
        x_pt,
        false,
        true,
        ndim_one,
        batch_ndim_none,
        gemm_alpha);

    auto grad_output = torch::full(out_pt.sizes(),
        static_cast<float>(grad_fill_val),
        torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false));
    out_pt.backward(grad_output);

    require_relative_frobenius_error(
        runtime.get_output<float>(c), out_pt, pytorch_gemm_tolerance);
    require_relative_frobenius_error(
        runtime.get_output<float>(w->grad()), w_pt.grad(), pytorch_gemm_tolerance);
    require_relative_frobenius_error(
        runtime.get_output<float>(x->grad()), x_pt.grad(), pytorch_gemm_tolerance);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph gemm attention output pattern matches PyTorch",
    "[graph][nn_graph][pytorch]")
{
    constexpr Index batch = 2;
    constexpr Index seq = 8;
    constexpr Index hidden = 64;
    constexpr Index head_size = 16;
    constexpr Index n_heads = 4;
    constexpr Scalar gemm_alpha = 1.0;

    const Index w_nelems = head_size * n_heads * hidden;
    const Index attn_nelems = batch * seq * head_size * n_heads;
    std::vector<float> w_data(static_cast<size_t>(w_nelems));
    std::vector<float> attn_data(static_cast<size_t>(attn_nelems));
    for (Index i = 0; i < w_nelems; ++i)
        w_data[static_cast<size_t>(i)] = 0.1f * static_cast<float>(i + 1);
    for (Index i = 0; i < attn_nelems; ++i)
        attn_data[static_cast<size_t>(i)] = 0.15f * static_cast<float>(i + 2);

    NNGraph g("gemm_attn_out");
    auto *w = g.tensor({head_size, n_heads, hidden}, DataType::FP32, true)
                  ->set_name("w");
    auto *attn = g.tensor({batch, seq, head_size, n_heads}, DataType::FP32, true)
                   ->set_name("attn");
    auto *c = gemm(w, attn, gemm_alpha, false, true, ndim_two, batch_ndim_none);

    w->mark_input(true);
    attn->mark_input(true);
    c->mark_output(true);

    TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());
    Runtime runtime(tile_graph);
    runtime.compile();
    runtime.bind_data(w, w_data);
    runtime.bind_data(attn, attn_data);
    runtime.execute();
    runtime.wait();

    auto w_pt = torch::from_blob(w_data.data(),
        {head_size, n_heads, hidden},
        torch::TensorOptions().dtype(torch::kFloat32))
                    .clone();
    auto attn_pt = torch::from_blob(attn_data.data(),
        {batch, seq, head_size, n_heads},
        torch::TensorOptions().dtype(torch::kFloat32))
                       .clone();
    auto out_pt = pytorch_gemm_reference(w_pt,
        attn_pt,
        false,
        true,
        ndim_two,
        batch_ndim_none,
        gemm_alpha);

    require_relative_frobenius_error(
        runtime.get_output<float>(c), out_pt, pytorch_gemm_tolerance);
}

#endif // NNTILE_HAVE_TORCH
