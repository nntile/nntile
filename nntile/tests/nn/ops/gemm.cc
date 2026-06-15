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

#ifdef NNTILE_HAVE_TORCH
#include "pytorch_helper.hh"
#include "pytorch_tile_helpers.hh"
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
    auto *x = g.tensor({N, K}, DataType::FP32)->set_name("x");
    auto *w = g.tensor({K, M}, DataType::FP32)->set_name("w");
    auto *c = gemm(x,
        w,
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
        auto *x = g.tensor({N2, N1, K2, K1}, DataType::FP32)->set_name("x");
        auto *w = g.tensor({K2, K1, M2, M1}, DataType::FP32)->set_name("w");
        auto *c = gemm(x,
            w,
            gemm_alpha_one,
            trans_a_default,
            trans_b_default,
            ndim_two,
            batch_ndim_none);
        REQUIRE(c != nullptr);
        REQUIRE(c->has_producer());
        REQUIRE(c->shape() == (std::vector<Index>{N2, N1, M2, M1}));
        REQUIRE(g.num_ops() == 1);
    }
    SECTION("ndim=1, batch_ndim=1: batched 2D matrices")
    {
        const Index B = 4, M = 2, K = 3, N = 5;
        NNGraph g("gemm_batched");
        auto *x = g.tensor({B, N, K}, DataType::FP32)->set_name("x");
        auto *w = g.tensor({B, K, M}, DataType::FP32)->set_name("w");
        auto *c = gemm(x,
            w,
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
    SECTION("ndim=2, batch_ndim=0: x.ndim() != w.ndim() (3D @ 4D)")
    {
        const Index M1 = 2, K1 = 3, K2 = 4, N1 = 5, N2 = 6;
        NNGraph g("gemm_3d_4d");
        auto *x = g.tensor({N2, N1, K2, K1}, DataType::FP32)->set_name("x");
        auto *w = g.tensor({K2, K1, M1}, DataType::FP32)->set_name("w");
        auto *c = gemm(x,
            w,
            gemm_alpha_one,
            trans_a_default,
            trans_b_default,
            ndim_two,
            batch_ndim_none);
        REQUIRE(c != nullptr);
        REQUIRE(c->has_producer());
        REQUIRE(x->ndim() == 4);
        REQUIRE(w->ndim() == 3);
        REQUIRE(c->shape() == (std::vector<Index>{N2, N1, M1}));
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
    auto *x = g.tensor({N, K}, DataType::FP32)->set_name("x");
    auto *w = g.tensor({K, M}, DataType::FP32)->set_name("w");
    auto *c = gemm(x,
        w,
        gemm_alpha_one,
        trans_a_default,
        trans_b_default,
        ndim_one,
        batch_ndim_none);

    auto [c_grad, _] = g.get_or_create_grad(c, "c_grad");
    gt::fill(grad_fill_val, c_grad->data());
    c->backward();

    REQUIRE(x->has_grad());
    REQUIRE(w->has_grad());
    REQUIRE(x->grad()->shape() == (std::vector<Index>{N, K}));
    REQUIRE(w->grad()->shape() == (std::vector<Index>{K, M}));
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph gemm backward multi-dimensional",
    "[graph][nn_graph]")
{
    SECTION("ndim=2, batch_ndim=0")
    {
        const Index M1 = 1, M2 = 2, K1 = 3, K2 = 4, N1 = 5, N2 = 6;
        NNGraph g("gemm_bwd_4d");
        auto *x = g.tensor({N2, N1, K2, K1}, DataType::FP32)->set_name("x");
        auto *w = g.tensor({K2, K1, M2, M1}, DataType::FP32)->set_name("w");
        auto *c = gemm(x,
            w,
            gemm_alpha_one,
            trans_a_default,
            trans_b_default,
            ndim_two,
            batch_ndim_none);
        auto [c_grad, _] = g.get_or_create_grad(c, "c_grad");
        gt::fill(Scalar(1.0), c_grad->data());
        c->backward();
        REQUIRE(x->has_grad());
        REQUIRE(w->has_grad());
        REQUIRE(x->grad()->shape() == (std::vector<Index>{N2, N1, K2, K1}));
        REQUIRE(w->grad()->shape() == (std::vector<Index>{K2, K1, M2, M1}));
    }
    SECTION("ndim=1, batch_ndim=1")
    {
        const Index B = 3, M = 2, K = 4, N = 3;
        NNGraph g("gemm_bwd_batched");
        auto *x = g.tensor({B, N, K}, DataType::FP32)->set_name("x");
        auto *w = g.tensor({B, K, M}, DataType::FP32)->set_name("w");
        auto *c = gemm(x,
            w,
            gemm_alpha_one,
            trans_a_default,
            trans_b_default,
            ndim_one,
            batch_ndim_one);
        auto [c_grad, _] = g.get_or_create_grad(c, "c_grad");
        gt::fill(Scalar(-1.0), c_grad->data());
        c->backward();
        REQUIRE(x->has_grad());
        REQUIRE(w->has_grad());
        REQUIRE(x->grad()->shape() == (std::vector<Index>{B, N, K}));
        REQUIRE(w->grad()->shape() == (std::vector<Index>{B, K, M}));
    }
    SECTION("ndim=2, batch_ndim=0: a.ndim() != b.ndim() (3D @ 4D)")
    {
        const Index M1 = 2, K1 = 3, K2 = 4, N1 = 5, N2 = 6;
        NNGraph g("gemm_bwd_3d_4d");
        auto *x = g.tensor({N2, N1, K2, K1}, DataType::FP32)->set_name("x");
        auto *w = g.tensor({K2, K1, M1}, DataType::FP32)->set_name("w");
        auto *c = gemm(x,
            w,
            gemm_alpha_one,
            trans_a_default,
            trans_b_default,
            ndim_two,
            batch_ndim_none);
        auto [c_grad, _] = g.get_or_create_grad(c, "c_grad");
        gt::fill(Scalar(1.0), c_grad->data());
        c->backward();
        REQUIRE(x->has_grad());
        REQUIRE(w->has_grad());
        REQUIRE(x->grad()->shape() == (std::vector<Index>{N2, N1, K2, K1}));
        REQUIRE(w->grad()->shape() == (std::vector<Index>{K2, K1, M1}));
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
    auto *x = g.tensor({N, K}, DataType::FP32, true)->set_name("x");
    auto *w = g.tensor({K, M}, DataType::FP32, true)->set_name("w");
    auto *c = gemm(x,
        w,
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

    REQUIRE(x->has_grad());
    REQUIRE(w->has_grad());
    REQUIRE(x->grad()->shape() == (std::vector<Index>{N, K}));
    REQUIRE(w->grad()->shape() == (std::vector<Index>{K, M}));
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph gemm forward and backward multi-dimensional",
    "[graph][nn_graph]")
{
    const auto [x_shape,
        w_shape,
        expected_c_shape,
        ndim,
        batch_ndim,
        alpha,
        grad_val] = GENERATE(std::tuple{std::vector<Index>{5, 3, 2, 4},
                                 std::vector<Index>{2, 4, 3, 2},
                                 std::vector<Index>{5, 3, 3, 2},
                                 ndim_two,
                                 batch_ndim_none,
                                 Scalar(1.0),
                                 Scalar(1.0)},
        std::tuple{std::vector<Index>{3, 3, 4},
            std::vector<Index>{3, 4, 2},
            std::vector<Index>{3, 3, 2},
            ndim_one,
            batch_ndim_one,
            Scalar(0.5),
            Scalar(-1.0)},
        std::tuple{std::vector<Index>{6, 5, 4, 3},
            std::vector<Index>{4, 3, 2},
            std::vector<Index>{6, 5, 2},
            ndim_two,
            batch_ndim_none,
            Scalar(1.0),
            Scalar(1.0)});

    NNGraph g("gemm_md");
    auto *x = g.tensor(x_shape, DataType::FP32, true)->set_name("x");
    auto *w = g.tensor(w_shape, DataType::FP32, true)->set_name("w");
    auto *c =
        gemm(x, w, alpha, trans_a_default, trans_b_default, ndim, batch_ndim);

    REQUIRE(c != nullptr);
    REQUIRE(c->shape() == expected_c_shape);

    auto [c_grad, _] = g.get_or_create_grad(c, "c_grad");
    gt::fill(grad_val, c_grad->data());
    c->backward();

    REQUIRE(x->has_grad());
    REQUIRE(w->has_grad());
    REQUIRE(x->grad()->shape() == x_shape);
    REQUIRE(w->grad()->shape() == w_shape);
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
        auto *x = g.tensor({3, 2}, DataType::FP32, true)->set_name("x");
        auto *w = g.tensor({4, 2}, DataType::FP32, true)->set_name("w");
        auto *c = gemm(
            x, w, gemm_alpha_one, trans_a, trans_b, ndim_one, batch_ndim_none);

        REQUIRE(c != nullptr);
        REQUIRE(c->shape() == (std::vector<Index>{3, 4}));

        auto [c_grad, _] = g.get_or_create_grad(c, "c_grad");
        gt::fill(grad_fill_val, c_grad->data());

        REQUIRE_NOTHROW(c->backward());
        REQUIRE(x->has_grad());
        REQUIRE(w->has_grad());
        REQUIRE(x->grad()->shape() == (std::vector<Index>{3, 2}));
        REQUIRE(w->grad()->shape() == (std::vector<Index>{4, 2}));
    }

    SECTION("trans_b=true uses grad_C transpose for grad_A")
    {
        constexpr bool trans_a = true;
        constexpr bool trans_b = true;

        NNGraph g("gemm_transposed_a_transposed_b");
        auto *x = g.tensor({2, 3}, DataType::FP32, true)->set_name("x");
        auto *w = g.tensor({4, 2}, DataType::FP32, true)->set_name("w");
        auto *c = gemm(
            x, w, gemm_alpha_one, trans_a, trans_b, ndim_one, batch_ndim_none);

        REQUIRE(c != nullptr);
        REQUIRE(c->shape() == (std::vector<Index>{3, 4}));

        auto [c_grad, _] = g.get_or_create_grad(c, "c_grad");
        gt::fill(grad_fill_val, c_grad->data());

        REQUIRE_NOTHROW(c->backward());
        REQUIRE(x->has_grad());
        REQUIRE(w->has_grad());
        REQUIRE(x->grad()->shape() == (std::vector<Index>{2, 3}));
        REQUIRE(w->grad()->shape() == (std::vector<Index>{4, 2}));
    }
}

#ifdef NNTILE_HAVE_TORCH

namespace
{

//! Heterogeneous splits for each extent (>=2 axes get multi-tile layouts).
std::vector<Index> nn_gemm_het_axis(Index extent)
{
    if (extent <= 1)
    {
        return {extent};
    }
    if (extent == 2)
    {
        return {1, 1};
    }
    if (extent == 3)
    {
        return {1, 2};
    }
    if (extent == 4)
    {
        return {1, 3};
    }
    if (extent == 5)
    {
        return {3, 2};
    }
    return {2, 3, extent - 5};
}

void nn_pytorch_tile_gemm_4d_operands(NNGraph::TensorNode *a,
    NNGraph::TensorNode *b,
    Index m1,
    Index m2,
    Index k1,
    Index k2,
    Index n1,
    Index n2)
{
    a->data()->axis(0)->set_tiling(nn_gemm_het_axis(m1));
    a->data()->axis(1)->set_tiling(nn_gemm_het_axis(m2));
    a->data()->axis(2)->set_tiling(nn_gemm_het_axis(k1));
    a->data()->axis(3)->set_tiling(nn_gemm_het_axis(k2));
    b->data()->axis(2)->set_tiling(nn_gemm_het_axis(n1));
    b->data()->axis(3)->set_tiling(nn_gemm_het_axis(n2));
}

void nn_pytorch_tile_gemm_batched_operands(NNGraph::TensorNode *a,
    NNGraph::TensorNode *b,
    Index M,
    Index K,
    Index N,
    Index B)
{
    a->data()->axis(0)->set_tiling(nn_gemm_het_axis(M));
    a->data()->axis(1)->set_tiling(nn_gemm_het_axis(K));
    a->data()->axis(2)->set_tiling(nn_gemm_het_axis(B));
    b->data()->axis(1)->set_tiling(nn_gemm_het_axis(N));
    b->data()->axis(2)->set_tiling(nn_gemm_het_axis(B));
}

} // namespace

using nntile::test::nn_pytorch_tile_gemm_operands_6_7_6;
using nntile::test::require_relative_frobenius_error;

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph gemm forward matches PyTorch",
    "[graph][nn_graph][pytorch]")
{
    const auto gemm_alpha = GENERATE(Scalar(1.0), Scalar(0.5), Scalar(2.0));
    constexpr Index M = 6;
    constexpr Index K = 7;
    constexpr Index N = 6;

    const Index x_nelems = N * K;
    const Index w_nelems = K * M;
    const Index c_nelems = N * M;

    std::vector<float> x_data(x_nelems);
    std::vector<float> w_data(w_nelems);
    for (Index i = 0; i < x_nelems; ++i)
        x_data[i] = 0.1f * static_cast<float>(i + 1);
    for (Index i = 0; i < w_nelems; ++i)
        w_data[i] = 0.15f * static_cast<float>(i + 2);

    NNGraph g("gemm_pytorch");
    auto *x = g.tensor({N, K}, DataType::FP32, true)->set_name("x");
    auto *w = g.tensor({K, M}, DataType::FP32, true)->set_name("w");
    auto *c = gemm(x,
        w,
        gemm_alpha,
        trans_a_default,
        trans_b_default,
        ndim_one,
        batch_ndim_none);

    nn_pytorch_tile_gemm_operands_6_7_6(w, x);

    x->mark_input(true);
    w->mark_input(true);
    c->mark_output(true);

    TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());
    Runtime runtime(tile_graph);
    runtime.compile();
    runtime.bind_data(x, x_data);
    runtime.bind_data(w, w_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> nntile_out = runtime.get_output<float>(c);

    auto x_pt = torch::from_blob(x_data.data(),
        {N, K},
        torch::TensorOptions().dtype(torch::kFloat32))
                    .clone()
                    .set_requires_grad(false);
    auto w_pt = torch::from_blob(w_data.data(),
        {K, M},
        torch::TensorOptions().dtype(torch::kFloat32))
                    .clone()
                    .set_requires_grad(false);
    auto out_pt =
        (gemm_alpha * torch::einsum("nk,km->nm", {x_pt, w_pt})).contiguous();

    require_relative_frobenius_error(nntile_out, out_pt);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph gemm backward matches PyTorch",
    "[graph][nn_graph][pytorch]")
{
    const auto [gemm_alpha, grad_fill_val] =
        GENERATE(std::tuple{Scalar(1.0), Scalar(1.0)},
            std::tuple{Scalar(0.5), Scalar(1.0)},
            std::tuple{Scalar(2.0), Scalar(-1.0)});
    constexpr Index M = 6;
    constexpr Index K = 7;
    constexpr Index N = 6;

    const Index x_nelems = N * K;
    const Index w_nelems = K * M;

    std::vector<float> x_data(x_nelems);
    std::vector<float> w_data(w_nelems);
    for (Index i = 0; i < x_nelems; ++i)
        x_data[i] = 0.1f * static_cast<float>(i + 1);
    for (Index i = 0; i < w_nelems; ++i)
        w_data[i] = 0.15f * static_cast<float>(i + 2);

    NNGraph g("gemm_bwd_pytorch");
    auto *x = g.tensor({N, K}, DataType::FP32, true)->set_name("x");
    auto *w = g.tensor({K, M}, DataType::FP32, true)->set_name("w");
    auto *c = gemm(x,
        w,
        gemm_alpha,
        trans_a_default,
        trans_b_default,
        ndim_one,
        batch_ndim_none);

    nn_pytorch_tile_gemm_operands_6_7_6(w, x);

    x->mark_input(true);
    w->mark_input(true);

    auto [c_grad, _] = g.get_or_create_grad(c, "c_grad");
    gt::fill(grad_fill_val, c_grad->data());
    c->backward();

    x->grad()->mark_output(true);
    w->grad()->mark_output(true);

    TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());
    Runtime runtime(tile_graph);
    runtime.compile();
    runtime.bind_data(x, x_data);
    runtime.bind_data(w, w_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> nntile_grad_x = runtime.get_output<float>(x->grad());
    std::vector<float> nntile_grad_w = runtime.get_output<float>(w->grad());

    auto x_pt = torch::from_blob(x_data.data(),
        {N, K},
        torch::TensorOptions().dtype(torch::kFloat32))
                    .clone()
                    .set_requires_grad(true);
    auto w_pt = torch::from_blob(w_data.data(),
        {K, M},
        torch::TensorOptions().dtype(torch::kFloat32))
                    .clone()
                    .set_requires_grad(true);
    auto out_pt = gemm_alpha * torch::einsum("nk,km->nm", {x_pt, w_pt});

    auto grad_output = torch::full({N, M},
        static_cast<float>(grad_fill_val),
        torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false));
    out_pt.backward(grad_output);

    require_relative_frobenius_error(nntile_grad_x, x_pt.grad());
    require_relative_frobenius_error(nntile_grad_w, w_pt.grad());
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph gemm forward matches PyTorch 4D",
    "[graph][nn_graph][pytorch]")
{
    const auto [M1, M2, K1, K2, N1, N2, gemm_alpha] =
        GENERATE(std::tuple{Index(2),
                     Index(2),
                     Index(3),
                     Index(2),
                     Index(2),
                     Index(3),
                     Scalar(1.0)},
            std::tuple{Index(2),
                Index(3),
                Index(4),
                Index(2),
                Index(3),
                Index(5),
                Scalar(0.5)});

    const Index x_nelems = N2 * N1 * K2 * K1;
    const Index w_nelems = M1 * M2 * K1 * K2;
    const Index c_nelems = M1 * M2 * N1 * N2;

    std::vector<float> x_data(x_nelems);
    std::vector<float> w_data(w_nelems);
    for (Index i = 0; i < x_nelems; ++i)
        x_data[i] = 0.1f * static_cast<float>(i + 1);
    for (Index i = 0; i < w_nelems; ++i)
        w_data[i] = 0.15f * static_cast<float>(i + 2);

    NNGraph g("gemm_pytorch_4d");
    auto *x = g.tensor({N2, N1, K2, K1}, DataType::FP32, true)->set_name("x");
    auto *w = g.tensor({K2, K1, M2, M1}, DataType::FP32, true)->set_name("w");
    auto *c = gemm(x,
        w,
        gemm_alpha,
        trans_a_default,
        trans_b_default,
        ndim_two,
        batch_ndim_none);

    nn_pytorch_tile_gemm_4d_operands(w, x, M1, M2, K1, K2, N1, N2);

    x->mark_input(true);
    w->mark_input(true);
    c->mark_output(true);

    TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());
    Runtime runtime(tile_graph);
    runtime.compile();
    runtime.bind_data(x, x_data);
    runtime.bind_data(w, w_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> nntile_out = runtime.get_output<float>(c);

    auto x_pt = torch::from_blob(x_data.data(),
        {N2, N1, K2, K1},
        torch::TensorOptions().dtype(torch::kFloat32))
                    .clone()
                    .set_requires_grad(false);
    auto w_pt = torch::from_blob(w_data.data(),
        {K2, K1, M2, M1},
        torch::TensorOptions().dtype(torch::kFloat32))
                    .clone()
                    .set_requires_grad(false);
    auto out_pt = (gemm_alpha *
                      torch::einsum("efwx,wxcd->efcd", {x_pt, w_pt}))
                      .contiguous();

    require_relative_frobenius_error(nntile_out, out_pt, 2e-5f);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph gemm forward matches PyTorch batched",
    "[graph][nn_graph][pytorch]")
{
    const auto [B, M, K, N, gemm_alpha] = GENERATE(
        std::tuple{Index(3), Index(2), Index(4), Index(3), Scalar(1.0)},
        std::tuple{Index(4), Index(3), Index(5), Index(2), Scalar(0.5)});

    const Index x_nelems = B * N * K;
    const Index w_nelems = B * K * M;
    const Index c_nelems = B * N * M;

    std::vector<float> x_data(x_nelems);
    std::vector<float> w_data(w_nelems);
    for (Index i = 0; i < x_nelems; ++i)
        x_data[i] = 0.1f * static_cast<float>(i + 1);
    for (Index i = 0; i < w_nelems; ++i)
        w_data[i] = 0.15f * static_cast<float>(i + 2);

    NNGraph g("gemm_pytorch_batched");
    auto *x = g.tensor({B, N, K}, DataType::FP32, true)->set_name("x");
    auto *w = g.tensor({B, K, M}, DataType::FP32, true)->set_name("w");
    auto *c = gemm(x,
        w,
        gemm_alpha,
        trans_a_default,
        trans_b_default,
        ndim_one,
        batch_ndim_one);

    nn_pytorch_tile_gemm_batched_operands(w, x, M, K, N, B);

    x->mark_input(true);
    w->mark_input(true);
    c->mark_output(true);

    TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());
    Runtime runtime(tile_graph);
    runtime.compile();
    runtime.bind_data(x, x_data);
    runtime.bind_data(w, w_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> nntile_out = runtime.get_output<float>(c);

    auto x_pt = torch::from_blob(x_data.data(),
        {B, N, K},
        torch::TensorOptions().dtype(torch::kFloat32))
                    .clone();
    auto w_pt = torch::from_blob(w_data.data(),
        {B, K, M},
        torch::TensorOptions().dtype(torch::kFloat32))
                    .clone();
    auto out_pt =
        (gemm_alpha * torch::einsum("bnk,bkm->bnm", {x_pt, w_pt})).contiguous();

    require_relative_frobenius_error(nntile_out, out_pt);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph gemm backward matches PyTorch 4D",
    "[graph][nn_graph][pytorch]")
{
    const auto [M1, M2, K1, K2, N1, N2, gemm_alpha, grad_fill_val] =
        GENERATE(std::tuple{Index(2),
                     Index(2),
                     Index(3),
                     Index(2),
                     Index(2),
                     Index(3),
                     Scalar(1.0),
                     Scalar(1.0)},
            std::tuple{Index(2),
                Index(3),
                Index(4),
                Index(2),
                Index(3),
                Index(5),
                Scalar(0.5),
                Scalar(-1.0)});

    const Index x_nelems = N2 * N1 * K2 * K1;
    const Index w_nelems = M1 * M2 * K1 * K2;

    std::vector<float> x_data(x_nelems);
    std::vector<float> w_data(w_nelems);
    for (Index i = 0; i < x_nelems; ++i)
        x_data[i] = 0.1f * static_cast<float>(i + 1);
    for (Index i = 0; i < w_nelems; ++i)
        w_data[i] = 0.15f * static_cast<float>(i + 2);

    NNGraph g("gemm_bwd_pytorch_4d");
    auto *x = g.tensor({N2, N1, K2, K1}, DataType::FP32, true)->set_name("x");
    auto *w = g.tensor({K2, K1, M2, M1}, DataType::FP32, true)->set_name("w");
    auto *c = gemm(x,
        w,
        gemm_alpha,
        trans_a_default,
        trans_b_default,
        ndim_two,
        batch_ndim_none);

    nn_pytorch_tile_gemm_4d_operands(w, x, M1, M2, K1, K2, N1, N2);

    x->mark_input(true);
    w->mark_input(true);

    auto [c_grad, _] = g.get_or_create_grad(c, "c_grad");
    gt::fill(grad_fill_val, c_grad->data());
    c->backward();

    x->grad()->mark_output(true);
    w->grad()->mark_output(true);

    TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());
    Runtime runtime(tile_graph);
    runtime.compile();
    runtime.bind_data(x, x_data);
    runtime.bind_data(w, w_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> nntile_grad_x =
        runtime.get_output<float>(x->grad());
    std::vector<float> nntile_grad_w =
        runtime.get_output<float>(w->grad());

    auto x_pt = torch::from_blob(x_data.data(),
        {N2, N1, K2, K1},
        torch::TensorOptions().dtype(torch::kFloat32))
                    .clone()
                    .set_requires_grad(true);
    auto w_pt = torch::from_blob(w_data.data(),
        {K2, K1, M2, M1},
        torch::TensorOptions().dtype(torch::kFloat32))
                    .clone()
                    .set_requires_grad(true);
    auto out_pt =
        gemm_alpha * torch::einsum("efwx,wxcd->efcd", {x_pt, w_pt});

    auto grad_out = torch::full({N2, N1, M2, M1},
        static_cast<float>(grad_fill_val),
        torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false));
    out_pt.backward(grad_out);

    require_relative_frobenius_error(nntile_grad_x, x_pt.grad());
    require_relative_frobenius_error(nntile_grad_w, w_pt.grad());
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph gemm backward matches PyTorch batched",
    "[graph][nn_graph][pytorch]")
{
    const auto [B, M, K, N, gemm_alpha, grad_fill_val] = GENERATE(
        std::tuple{
            Index(3), Index(2), Index(4), Index(3), Scalar(1.0), Scalar(1.0)},
        std::tuple{Index(4),
            Index(3),
            Index(5),
            Index(2),
            Scalar(0.5),
            Scalar(-1.0)});

    const Index x_nelems = B * N * K;
    const Index w_nelems = B * K * M;

    std::vector<float> x_data(x_nelems);
    std::vector<float> w_data(w_nelems);
    for (Index i = 0; i < x_nelems; ++i)
        x_data[i] = 0.1f * static_cast<float>(i + 1);
    for (Index i = 0; i < w_nelems; ++i)
        w_data[i] = 0.15f * static_cast<float>(i + 2);

    NNGraph g("gemm_bwd_pytorch_batched");
    auto *x = g.tensor({B, N, K}, DataType::FP32, true)->set_name("x");
    auto *w = g.tensor({B, K, M}, DataType::FP32, true)->set_name("w");
    auto *c = gemm(x,
        w,
        gemm_alpha,
        trans_a_default,
        trans_b_default,
        ndim_one,
        batch_ndim_one);

    nn_pytorch_tile_gemm_batched_operands(w, x, M, K, N, B);

    x->mark_input(true);
    w->mark_input(true);

    auto [c_grad, _] = g.get_or_create_grad(c, "c_grad");
    gt::fill(grad_fill_val, c_grad->data());
    c->backward();

    x->grad()->mark_output(true);
    w->grad()->mark_output(true);

    TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());
    Runtime runtime(tile_graph);
    runtime.compile();
    runtime.bind_data(x, x_data);
    runtime.bind_data(w, w_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> nntile_grad_x =
        runtime.get_output<float>(x->grad());
    std::vector<float> nntile_grad_w =
        runtime.get_output<float>(w->grad());

    auto x_pt = torch::from_blob(x_data.data(),
        {B, N, K},
        torch::TensorOptions().dtype(torch::kFloat32))
                    .clone()
                    .set_requires_grad(true);
    auto w_pt = torch::from_blob(w_data.data(),
        {B, K, M},
        torch::TensorOptions().dtype(torch::kFloat32))
                    .clone()
                    .set_requires_grad(true);
    auto out_pt = gemm_alpha * torch::einsum("bnk,bkm->bnm", {x_pt, w_pt});

    auto grad_output = torch::full({B, N, M},
        static_cast<float>(grad_fill_val),
        torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false));
    out_pt.backward(grad_output);

    require_relative_frobenius_error(nntile_grad_x, x_pt.grad());
    require_relative_frobenius_error(nntile_grad_w, w_pt.grad());
}

#endif // NNTILE_HAVE_TORCH
