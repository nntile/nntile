/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/nn/gemm.cc
 * Test NNGraph gemm autograd operation.
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

constexpr Scalar gemm_alpha_one = 1.0;
constexpr bool trans_a_default = false;
constexpr bool trans_b_default = false;
constexpr Index ndim_one = 1;
constexpr Index ndim_two = 2;
constexpr Index batch_ndim_none = 0;
constexpr Index batch_ndim_one = 1;

} // anonymous namespace

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph gemm structure", "[graph][nn_graph]")
{
    const auto [M, K, N] = GENERATE(
        std::tuple{Index(2), Index(3), Index(4)},
        std::tuple{Index(3), Index(4), Index(3)});

    NNGraph g("gemm_structure");
    auto* a = g.tensor({M, K}, "a", DataType::FP32);
    auto* b = g.tensor({K, N}, "b", DataType::FP32);
    auto* c = gemm(a, b, "c", gemm_alpha_one, trans_a_default, trans_b_default,
                   ndim_one, batch_ndim_none);

    REQUIRE(c != nullptr);
    REQUIRE(c->has_producer());
    REQUIRE(c->shape() == (std::vector<Index>{M, N}));
    REQUIRE(g.num_ops() == 1);
    REQUIRE(g.tensor_graph().ops()[0]->op_name() == "GEMM");
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph gemm structure multi-dimensional", "[graph][nn_graph]")
{
    SECTION("ndim=2, batch_ndim=0: 4D contraction")
    {
        const Index M1 = 2, M2 = 3, K1 = 4, K2 = 2, N1 = 3, N2 = 5;
        NNGraph g("gemm_4d");
        auto* a = g.tensor({M1, M2, K1, K2}, "a", DataType::FP32);
        auto* b = g.tensor({K1, K2, N1, N2}, "b", DataType::FP32);
        auto* c = gemm(a, b, "c", gemm_alpha_one, trans_a_default, trans_b_default,
                       ndim_two, batch_ndim_none);
        REQUIRE(c != nullptr);
        REQUIRE(c->has_producer());
        REQUIRE(c->shape() == (std::vector<Index>{M1, M2, N1, N2}));
        REQUIRE(g.num_ops() == 1);
    }
    SECTION("ndim=1, batch_ndim=1: batched 2D matrices")
    {
        const Index B = 4, M = 2, K = 3, N = 5;
        NNGraph g("gemm_batched");
        auto* a = g.tensor({M, K, B}, "a", DataType::FP32);
        auto* b = g.tensor({K, N, B}, "b", DataType::FP32);
        auto* c = gemm(a, b, "c", gemm_alpha_one, trans_a_default, trans_b_default,
                       ndim_one, batch_ndim_one);
        REQUIRE(c != nullptr);
        REQUIRE(c->has_producer());
        REQUIRE(c->shape() == (std::vector<Index>{M, N, B}));
        REQUIRE(g.num_ops() == 1);
    }
    SECTION("ndim=2, batch_ndim=0: a.ndim() != b.ndim() (3D @ 4D)")
    {
        const Index M1 = 2, K1 = 3, K2 = 4, N1 = 5, N2 = 6;
        NNGraph g("gemm_3d_4d");
        auto* a = g.tensor({M1, K1, K2}, "a", DataType::FP32);
        auto* b = g.tensor({K1, K2, N1, N2}, "b", DataType::FP32);
        auto* c = gemm(a, b, "c", gemm_alpha_one, trans_a_default, trans_b_default,
                       ndim_two, batch_ndim_none);
        REQUIRE(c != nullptr);
        REQUIRE(c->has_producer());
        REQUIRE(a->ndim() == 3);
        REQUIRE(b->ndim() == 4);
        REQUIRE(c->shape() == (std::vector<Index>{M1, N1, N2}));
        REQUIRE(g.num_ops() == 1);
    }
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph gemm backward", "[graph][nn_graph]")
{
    const auto [M, K, N, grad_fill_val] = GENERATE(
        std::tuple{Index(2), Index(3), Index(4), Scalar(1.0)},
        std::tuple{Index(3), Index(4), Index(3), Scalar(-1.0)});

    NNGraph g("gemm_backward");
    auto* a = g.tensor({M, K}, "a", DataType::FP32);
    auto* b = g.tensor({K, N}, "b", DataType::FP32);
    auto* c = gemm(a, b, "c", gemm_alpha_one, trans_a_default, trans_b_default,
                   ndim_one, batch_ndim_none);

    auto [c_grad, _] = g.get_or_create_grad(c, "c_grad");
    fill(grad_fill_val, c_grad->data());
    c->backward();

    REQUIRE(a->has_grad());
    REQUIRE(b->has_grad());
    REQUIRE(a->grad()->shape() == (std::vector<Index>{M, K}));
    REQUIRE(b->grad()->shape() == (std::vector<Index>{K, N}));
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph gemm backward multi-dimensional", "[graph][nn_graph]")
{
    SECTION("ndim=2, batch_ndim=0")
    {
        const Index M1 = 1, M2 = 2, K1 = 3, K2 = 4, N1 = 5, N2 = 6;
        NNGraph g("gemm_bwd_4d");
        auto* a = g.tensor({M1, M2, K1, K2}, "a", DataType::FP32);
        auto* b = g.tensor({K1, K2, N1, N2}, "b", DataType::FP32);
        auto* c = gemm(a, b, "c", gemm_alpha_one, trans_a_default, trans_b_default,
                       ndim_two, batch_ndim_none);
        auto [c_grad, _] = g.get_or_create_grad(c, "c_grad");
        fill(Scalar(1.0), c_grad->data());
        c->backward();
        REQUIRE(a->has_grad());
        REQUIRE(b->has_grad());
        REQUIRE(a->grad()->shape() == (std::vector<Index>{M1, M2, K1, K2}));
        REQUIRE(b->grad()->shape() == (std::vector<Index>{K1, K2, N1, N2}));
    }
    SECTION("ndim=1, batch_ndim=1")
    {
        const Index B = 3, M = 2, K = 4, N = 3;
        NNGraph g("gemm_bwd_batched");
        auto* a = g.tensor({M, K, B}, "a", DataType::FP32);
        auto* b = g.tensor({K, N, B}, "b", DataType::FP32);
        auto* c = gemm(a, b, "c", gemm_alpha_one, trans_a_default, trans_b_default,
                       ndim_one, batch_ndim_one);
        auto [c_grad, _] = g.get_or_create_grad(c, "c_grad");
        fill(Scalar(-1.0), c_grad->data());
        c->backward();
        REQUIRE(a->has_grad());
        REQUIRE(b->has_grad());
        REQUIRE(a->grad()->shape() == (std::vector<Index>{M, K, B}));
        REQUIRE(b->grad()->shape() == (std::vector<Index>{K, N, B}));
    }
    SECTION("ndim=2, batch_ndim=0: a.ndim() != b.ndim() (3D @ 4D)")
    {
        const Index M1 = 2, K1 = 3, K2 = 4, N1 = 5, N2 = 6;
        NNGraph g("gemm_bwd_3d_4d");
        auto* a = g.tensor({M1, K1, K2}, "a", DataType::FP32);
        auto* b = g.tensor({K1, K2, N1, N2}, "b", DataType::FP32);
        auto* c = gemm(a, b, "c", gemm_alpha_one, trans_a_default, trans_b_default,
                       ndim_two, batch_ndim_none);
        auto [c_grad, _] = g.get_or_create_grad(c, "c_grad");
        fill(Scalar(1.0), c_grad->data());
        c->backward();
        REQUIRE(a->has_grad());
        REQUIRE(b->has_grad());
        REQUIRE(a->grad()->shape() == (std::vector<Index>{M1, K1, K2}));
        REQUIRE(b->grad()->shape() == (std::vector<Index>{K1, K2, N1, N2}));
    }
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph gemm forward and backward", "[graph][nn_graph]")
{
    const auto [M, K, N, gemm_alpha, grad_fill_val] = GENERATE(
        std::tuple{Index(2), Index(3), Index(4), Scalar(1.0), Scalar(1.0)},
        std::tuple{Index(3), Index(4), Index(3), Scalar(0.5), Scalar(1.0)},
        std::tuple{Index(4), Index(5), Index(6), Scalar(2.0), Scalar(-1.0)});

    NNGraph g("gemm");
    auto* a = g.tensor({M, K}, "a", DataType::FP32, true);
    auto* b = g.tensor({K, N}, "b", DataType::FP32, true);
    auto* c = gemm(a, b, "c", gemm_alpha, trans_a_default, trans_b_default,
                   ndim_one, batch_ndim_none);

    REQUIRE(c != nullptr);
    REQUIRE(c->has_producer());
    REQUIRE(c->shape() == (std::vector<Index>{M, N}));

    auto [c_grad, _] = g.get_or_create_grad(c, "c_grad");
    fill(grad_fill_val, c_grad->data());
    c->backward();

    REQUIRE(a->has_grad());
    REQUIRE(b->has_grad());
    REQUIRE(a->grad()->shape() == (std::vector<Index>{M, K}));
    REQUIRE(b->grad()->shape() == (std::vector<Index>{K, N}));
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph gemm forward and backward multi-dimensional", "[graph][nn_graph]")
{
    const auto [a_shape, b_shape, expected_c_shape, ndim, batch_ndim, alpha, grad_val] =
        GENERATE(
            std::tuple{std::vector<Index>{2, 3, 4, 2}, std::vector<Index>{4, 2, 3, 5},
                       std::vector<Index>{2, 3, 3, 5}, ndim_two, batch_ndim_none,
                       Scalar(1.0), Scalar(1.0)},
            std::tuple{std::vector<Index>{2, 4, 3}, std::vector<Index>{4, 3, 3},
                       std::vector<Index>{2, 3, 3}, ndim_one, batch_ndim_one,
                       Scalar(0.5), Scalar(-1.0)},
            std::tuple{std::vector<Index>{2, 3, 4}, std::vector<Index>{3, 4, 5, 6},
                       std::vector<Index>{2, 5, 6}, ndim_two, batch_ndim_none,
                       Scalar(1.0), Scalar(1.0)});

    NNGraph g("gemm_md");
    auto* a = g.tensor(a_shape, "a", DataType::FP32, true);
    auto* b = g.tensor(b_shape, "b", DataType::FP32, true);
    auto* c = gemm(a, b, "c", alpha, trans_a_default, trans_b_default,
                   ndim, batch_ndim);

    REQUIRE(c != nullptr);
    REQUIRE(c->shape() == expected_c_shape);

    auto [c_grad, _] = g.get_or_create_grad(c, "c_grad");
    fill(grad_val, c_grad->data());
    c->backward();

    REQUIRE(a->has_grad());
    REQUIRE(b->has_grad());
    REQUIRE(a->grad()->shape() == a_shape);
    REQUIRE(b->grad()->shape() == b_shape);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph gemm backward transposed A", "[graph][nn_graph]")
{
    constexpr Scalar grad_fill_val = 1.0;

    SECTION("trans_b=false uses grad_C transpose for grad_A")
    {
        constexpr bool trans_a = true;
        constexpr bool trans_b = false;

        NNGraph g("gemm_transposed_a");
        auto* a = g.tensor({4, 2}, "a", DataType::FP32, true);
        auto* b = g.tensor({4, 3}, "b", DataType::FP32, true);
        auto* c = gemm(a, b, "c", gemm_alpha_one, trans_a, trans_b,
                       ndim_one, batch_ndim_none);

        REQUIRE(c != nullptr);
        REQUIRE(c->shape() == (std::vector<Index>{2, 3}));

        auto [c_grad, _] = g.get_or_create_grad(c, "c_grad");
        fill(grad_fill_val, c_grad->data());

        REQUIRE_NOTHROW(c->backward());
        REQUIRE(a->has_grad());
        REQUIRE(b->has_grad());
        REQUIRE(a->grad()->shape() == (std::vector<Index>{4, 2}));
        REQUIRE(b->grad()->shape() == (std::vector<Index>{4, 3}));
    }

    SECTION("trans_b=true uses grad_C transpose for grad_A")
    {
        constexpr bool trans_a = true;
        constexpr bool trans_b = true;

        NNGraph g("gemm_transposed_a_transposed_b");
        auto* a = g.tensor({4, 2}, "a", DataType::FP32, true);
        auto* b = g.tensor({3, 4}, "b", DataType::FP32, true);
        auto* c = gemm(a, b, "c", gemm_alpha_one, trans_a, trans_b,
                       ndim_one, batch_ndim_none);

        REQUIRE(c != nullptr);
        REQUIRE(c->shape() == (std::vector<Index>{2, 3}));

        auto [c_grad, _] = g.get_or_create_grad(c, "c_grad");
        fill(grad_fill_val, c_grad->data());

        REQUIRE_NOTHROW(c->backward());
        REQUIRE(a->has_grad());
        REQUIRE(b->has_grad());
        REQUIRE(a->grad()->shape() == (std::vector<Index>{4, 2}));
        REQUIRE(b->grad()->shape() == (std::vector<Index>{3, 4}));
    }
}

#ifdef NNTILE_HAVE_TORCH

using nntile::test::colmajor_to_rowmajor;
using nntile::test::compare_float_vectors;
using nntile::test::permute_rowmajor;
using nntile::test::pytorch_tolerance;

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph gemm forward matches PyTorch", "[graph][nn_graph][pytorch]")
{
    const auto [M, K, N, gemm_alpha] = GENERATE(
        std::tuple{Index(2), Index(3), Index(4), Scalar(1.0)},
        std::tuple{Index(3), Index(4), Index(3), Scalar(0.5)},
        std::tuple{Index(4), Index(5), Index(6), Scalar(2.0)});

    const Index a_nelems = M * K;
    const Index b_nelems = K * N;
    const Index c_nelems = M * N;

    std::vector<float> a_data(a_nelems);
    std::vector<float> b_data(b_nelems);
    for(Index i = 0; i < a_nelems; ++i)
        a_data[i] = 0.1f * static_cast<float>(i + 1);
    for(Index i = 0; i < b_nelems; ++i)
        b_data[i] = 0.15f * static_cast<float>(i + 2);

    std::vector<float> a_rowmajor = colmajor_to_rowmajor(a_data, {M, K});
    std::vector<float> b_rowmajor = colmajor_to_rowmajor(b_data, {K, N});

    NNGraph g("gemm_pytorch");
    auto* a = g.tensor({M, K}, "a", DataType::FP32, true);
    auto* b = g.tensor({K, N}, "b", DataType::FP32, true);
    auto* c = gemm(a, b, "c", gemm_alpha, trans_a_default, trans_b_default,
                   ndim_one, batch_ndim_none);

    a->mark_input(true);
    b->mark_input(true);
    c->mark_output(true);

    TensorGraph::Runtime runtime(g.tensor_graph());
    runtime.compile();
    runtime.bind_data("a", a_data);
    runtime.bind_data("b", b_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> nntile_out_colmajor = runtime.get_output<float>("c");
    std::vector<float> nntile_out =
        colmajor_to_rowmajor(nntile_out_colmajor, {M, N});

    auto a_pt = torch::from_blob(a_rowmajor.data(), {M, K},
        torch::TensorOptions().dtype(torch::kFloat32)).clone().set_requires_grad(false);
    auto b_pt = torch::from_blob(b_rowmajor.data(), {K, N},
        torch::TensorOptions().dtype(torch::kFloat32)).clone().set_requires_grad(false);
    auto out_pt = (gemm_alpha * torch::mm(a_pt, b_pt)).contiguous();

    std::vector<float> pytorch_out(out_pt.data_ptr<float>(),
                                   out_pt.data_ptr<float>() + c_nelems);

    REQUIRE(nntile_out.size() == pytorch_out.size());
    for(size_t i = 0; i < nntile_out.size(); ++i)
        REQUIRE(std::abs(nntile_out[i] - pytorch_out[i]) < pytorch_tolerance);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph gemm backward matches PyTorch", "[graph][nn_graph][pytorch]")
{
    const auto [M, K, N, gemm_alpha, grad_fill_val] = GENERATE(
        std::tuple{Index(2), Index(3), Index(4), Scalar(1.0), Scalar(1.0)},
        std::tuple{Index(3), Index(4), Index(3), Scalar(0.5), Scalar(1.0)},
        std::tuple{Index(4), Index(5), Index(6), Scalar(2.0), Scalar(-1.0)});

    const Index a_nelems = M * K;
    const Index b_nelems = K * N;

    std::vector<float> a_data(a_nelems);
    std::vector<float> b_data(b_nelems);
    for(Index i = 0; i < a_nelems; ++i)
        a_data[i] = 0.1f * static_cast<float>(i + 1);
    for(Index i = 0; i < b_nelems; ++i)
        b_data[i] = 0.15f * static_cast<float>(i + 2);

    std::vector<float> a_rowmajor = colmajor_to_rowmajor(a_data, {M, K});
    std::vector<float> b_rowmajor = colmajor_to_rowmajor(b_data, {K, N});

    NNGraph g("gemm_bwd_pytorch");
    auto* a = g.tensor({M, K}, "a", DataType::FP32, true);
    auto* b = g.tensor({K, N}, "b", DataType::FP32, true);
    auto* c = gemm(a, b, "c", gemm_alpha, trans_a_default, trans_b_default,
                   ndim_one, batch_ndim_none);

    a->mark_input(true);
    b->mark_input(true);

    auto [c_grad, _] = g.get_or_create_grad(c, "c_grad");
    fill(grad_fill_val, c_grad->data());
    c->backward();

    a->grad()->mark_output(true);
    b->grad()->mark_output(true);

    TensorGraph::Runtime runtime(g.tensor_graph());
    runtime.compile();
    runtime.bind_data("a", a_data);
    runtime.bind_data("b", b_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> nntile_grad_a_colmajor =
        runtime.get_output<float>(a->grad()->name());
    std::vector<float> nntile_grad_b_colmajor =
        runtime.get_output<float>(b->grad()->name());
    std::vector<float> nntile_grad_a =
        colmajor_to_rowmajor(nntile_grad_a_colmajor, {M, K});
    std::vector<float> nntile_grad_b =
        colmajor_to_rowmajor(nntile_grad_b_colmajor, {K, N});

    auto a_pt = torch::from_blob(a_rowmajor.data(), {M, K},
        torch::TensorOptions().dtype(torch::kFloat32)).clone().set_requires_grad(true);
    auto b_pt = torch::from_blob(b_rowmajor.data(), {K, N},
        torch::TensorOptions().dtype(torch::kFloat32)).clone().set_requires_grad(true);
    auto out_pt = gemm_alpha * torch::mm(a_pt, b_pt);

    auto grad_output = torch::full({M, N}, static_cast<float>(grad_fill_val),
        torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false));
    out_pt.backward(grad_output);

    compare_float_vectors(nntile_grad_a, a_pt.grad());
    compare_float_vectors(nntile_grad_b, b_pt.grad());
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph gemm forward matches PyTorch 4D", "[graph][nn_graph][pytorch]")
{
    const auto [M1, M2, K1, K2, N1, N2, gemm_alpha] = GENERATE(
        std::tuple{Index(2), Index(2), Index(3), Index(2), Index(2), Index(3), Scalar(1.0)},
        std::tuple{Index(2), Index(3), Index(4), Index(2), Index(3), Index(5), Scalar(0.5)});

    const Index a_nelems = M1 * M2 * K1 * K2;
    const Index b_nelems = K1 * K2 * N1 * N2;
    const Index c_nelems = M1 * M2 * N1 * N2;

    std::vector<float> a_data(a_nelems);
    std::vector<float> b_data(b_nelems);
    for(Index i = 0; i < a_nelems; ++i)
        a_data[i] = 0.1f * static_cast<float>(i + 1);
    for(Index i = 0; i < b_nelems; ++i)
        b_data[i] = 0.15f * static_cast<float>(i + 2);

    std::vector<float> a_rowmajor = colmajor_to_rowmajor(a_data, {M1, M2, K1, K2});
    std::vector<float> b_rowmajor = colmajor_to_rowmajor(b_data, {K1, K2, N1, N2});

    NNGraph g("gemm_pytorch_4d");
    auto* a = g.tensor({M1, M2, K1, K2}, "a", DataType::FP32, true);
    auto* b = g.tensor({K1, K2, N1, N2}, "b", DataType::FP32, true);
    auto* c = gemm(a, b, "c", gemm_alpha, trans_a_default, trans_b_default,
                   ndim_two, batch_ndim_none);

    a->mark_input(true);
    b->mark_input(true);
    c->mark_output(true);

    TensorGraph::Runtime runtime(g.tensor_graph());
    runtime.compile();
    runtime.bind_data("a", a_data);
    runtime.bind_data("b", b_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> nntile_out_colmajor = runtime.get_output<float>("c");
    std::vector<float> nntile_out =
        colmajor_to_rowmajor(nntile_out_colmajor, {M1, M2, N1, N2});

    auto a_pt = torch::from_blob(a_rowmajor.data(), {M1, M2, K1, K2},
        torch::TensorOptions().dtype(torch::kFloat32)).clone().set_requires_grad(false);
    auto b_pt = torch::from_blob(b_rowmajor.data(), {K1, K2, N1, N2},
        torch::TensorOptions().dtype(torch::kFloat32)).clone().set_requires_grad(false);
    auto a_2d = a_pt.reshape({M1 * M2, K1 * K2});
    auto b_2d = b_pt.reshape({K1 * K2, N1 * N2});
    auto out_pt = (gemm_alpha * torch::mm(a_2d, b_2d)).reshape({M1, M2, N1, N2}).contiguous();

    std::vector<float> pytorch_out(out_pt.data_ptr<float>(),
                                   out_pt.data_ptr<float>() + c_nelems);

    REQUIRE(nntile_out.size() == pytorch_out.size());
    const float tol_4d = 2e-5f;
    for(size_t i = 0; i < nntile_out.size(); ++i)
        REQUIRE(std::abs(nntile_out[i] - pytorch_out[i]) < tol_4d);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph gemm forward matches PyTorch batched", "[graph][nn_graph][pytorch]")
{
    const auto [B, M, K, N, gemm_alpha] = GENERATE(
        std::tuple{Index(3), Index(2), Index(4), Index(3), Scalar(1.0)},
        std::tuple{Index(4), Index(3), Index(5), Index(2), Scalar(0.5)});

    const Index a_nelems = M * K * B;
    const Index b_nelems = K * N * B;
    const Index c_nelems = M * N * B;

    std::vector<float> a_data(a_nelems);
    std::vector<float> b_data(b_nelems);
    for(Index i = 0; i < a_nelems; ++i)
        a_data[i] = 0.1f * static_cast<float>(i + 1);
    for(Index i = 0; i < b_nelems; ++i)
        b_data[i] = 0.15f * static_cast<float>(i + 2);

    std::vector<float> a_rowmajor = colmajor_to_rowmajor(a_data, {M, K, B});
    std::vector<float> b_rowmajor = colmajor_to_rowmajor(b_data, {K, N, B});

    NNGraph g("gemm_pytorch_batched");
    auto* a = g.tensor({M, K, B}, "a", DataType::FP32, true);
    auto* b = g.tensor({K, N, B}, "b", DataType::FP32, true);
    auto* c = gemm(a, b, "c", gemm_alpha, trans_a_default, trans_b_default,
                   ndim_one, batch_ndim_one);

    a->mark_input(true);
    b->mark_input(true);
    c->mark_output(true);

    TensorGraph::Runtime runtime(g.tensor_graph());
    runtime.compile();
    runtime.bind_data("a", a_data);
    runtime.bind_data("b", b_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> nntile_out_colmajor = runtime.get_output<float>("c");
    std::vector<float> nntile_out_rowmajor =
        colmajor_to_rowmajor(nntile_out_colmajor, {M, N, B});
    std::vector<float> nntile_out =
        permute_rowmajor(nntile_out_rowmajor, {M, N, B}, {2, 0, 1});

    auto a_pt = torch::from_blob(a_rowmajor.data(), {M, K, B},
        torch::TensorOptions().dtype(torch::kFloat32)).clone();
    auto b_pt = torch::from_blob(b_rowmajor.data(), {K, N, B},
        torch::TensorOptions().dtype(torch::kFloat32)).clone();
    auto a_batched = a_pt.permute({2, 0, 1});
    auto b_batched = b_pt.permute({2, 0, 1});
    auto out_pt = (gemm_alpha * torch::bmm(a_batched, b_batched)).contiguous();

    std::vector<float> pytorch_out(out_pt.data_ptr<float>(),
                                   out_pt.data_ptr<float>() + c_nelems);

    REQUIRE(nntile_out.size() == pytorch_out.size());
    for(size_t i = 0; i < nntile_out.size(); ++i)
        REQUIRE(std::abs(nntile_out[i] - pytorch_out[i]) < pytorch_tolerance);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph gemm backward matches PyTorch 4D", "[graph][nn_graph][pytorch]")
{
    const auto [M1, M2, K1, K2, N1, N2, gemm_alpha, grad_fill_val] = GENERATE(
        std::tuple{Index(2), Index(2), Index(3), Index(2), Index(2), Index(3),
                   Scalar(1.0), Scalar(1.0)},
        std::tuple{Index(2), Index(3), Index(4), Index(2), Index(3), Index(5),
                   Scalar(0.5), Scalar(-1.0)});

    const Index a_nelems = M1 * M2 * K1 * K2;
    const Index b_nelems = K1 * K2 * N1 * N2;

    std::vector<float> a_data(a_nelems);
    std::vector<float> b_data(b_nelems);
    for(Index i = 0; i < a_nelems; ++i)
        a_data[i] = 0.1f * static_cast<float>(i + 1);
    for(Index i = 0; i < b_nelems; ++i)
        b_data[i] = 0.15f * static_cast<float>(i + 2);

    std::vector<float> a_rowmajor = colmajor_to_rowmajor(a_data, {M1, M2, K1, K2});
    std::vector<float> b_rowmajor = colmajor_to_rowmajor(b_data, {K1, K2, N1, N2});

    NNGraph g("gemm_bwd_pytorch_4d");
    auto* a = g.tensor({M1, M2, K1, K2}, "a", DataType::FP32, true);
    auto* b = g.tensor({K1, K2, N1, N2}, "b", DataType::FP32, true);
    auto* c = gemm(a, b, "c", gemm_alpha, trans_a_default, trans_b_default,
                   ndim_two, batch_ndim_none);

    a->mark_input(true);
    b->mark_input(true);

    auto [c_grad, _] = g.get_or_create_grad(c, "c_grad");
    fill(grad_fill_val, c_grad->data());
    c->backward();

    a->grad()->mark_output(true);
    b->grad()->mark_output(true);

    TensorGraph::Runtime runtime(g.tensor_graph());
    runtime.compile();
    runtime.bind_data("a", a_data);
    runtime.bind_data("b", b_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> nntile_grad_a =
        colmajor_to_rowmajor(runtime.get_output<float>(a->grad()->name()),
                             {M1, M2, K1, K2});
    std::vector<float> nntile_grad_b =
        colmajor_to_rowmajor(runtime.get_output<float>(b->grad()->name()),
                             {K1, K2, N1, N2});

    auto a_pt = torch::from_blob(a_rowmajor.data(), {M1, M2, K1, K2},
        torch::TensorOptions().dtype(torch::kFloat32)).clone().set_requires_grad(true);
    auto b_pt = torch::from_blob(b_rowmajor.data(), {K1, K2, N1, N2},
        torch::TensorOptions().dtype(torch::kFloat32)).clone().set_requires_grad(true);
    auto a_2d = a_pt.reshape({M1 * M2, K1 * K2});
    auto b_2d = b_pt.reshape({K1 * K2, N1 * N2});
    auto out_pt = gemm_alpha * torch::mm(a_2d, b_2d);

    auto grad_2d = torch::full({M1 * M2, N1 * N2}, static_cast<float>(grad_fill_val),
        torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false));
    out_pt.backward(grad_2d);

    compare_float_vectors(nntile_grad_a, a_pt.grad().reshape({M1, M2, K1, K2}));
    compare_float_vectors(nntile_grad_b, b_pt.grad().reshape({K1, K2, N1, N2}));
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph gemm backward matches PyTorch batched", "[graph][nn_graph][pytorch]")
{
    const auto [B, M, K, N, gemm_alpha, grad_fill_val] = GENERATE(
        std::tuple{Index(3), Index(2), Index(4), Index(3), Scalar(1.0), Scalar(1.0)},
        std::tuple{Index(4), Index(3), Index(5), Index(2), Scalar(0.5), Scalar(-1.0)});

    const Index a_nelems = M * K * B;
    const Index b_nelems = K * N * B;

    std::vector<float> a_data(a_nelems);
    std::vector<float> b_data(b_nelems);
    for(Index i = 0; i < a_nelems; ++i)
        a_data[i] = 0.1f * static_cast<float>(i + 1);
    for(Index i = 0; i < b_nelems; ++i)
        b_data[i] = 0.15f * static_cast<float>(i + 2);

    std::vector<float> a_rowmajor = colmajor_to_rowmajor(a_data, {M, K, B});
    std::vector<float> b_rowmajor = colmajor_to_rowmajor(b_data, {K, N, B});

    NNGraph g("gemm_bwd_pytorch_batched");
    auto* a = g.tensor({M, K, B}, "a", DataType::FP32, true);
    auto* b = g.tensor({K, N, B}, "b", DataType::FP32, true);
    auto* c = gemm(a, b, "c", gemm_alpha, trans_a_default, trans_b_default,
                   ndim_one, batch_ndim_one);

    a->mark_input(true);
    b->mark_input(true);

    auto [c_grad, _] = g.get_or_create_grad(c, "c_grad");
    fill(grad_fill_val, c_grad->data());
    c->backward();

    a->grad()->mark_output(true);
    b->grad()->mark_output(true);

    TensorGraph::Runtime runtime(g.tensor_graph());
    runtime.compile();
    runtime.bind_data("a", a_data);
    runtime.bind_data("b", b_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> nntile_grad_a_rowmajor =
        colmajor_to_rowmajor(runtime.get_output<float>(a->grad()->name()),
                             {M, K, B});
    std::vector<float> nntile_grad_b_rowmajor =
        colmajor_to_rowmajor(runtime.get_output<float>(b->grad()->name()),
                             {K, N, B});
    std::vector<float> nntile_grad_a =
        permute_rowmajor(nntile_grad_a_rowmajor, {M, K, B}, {2, 0, 1});
    std::vector<float> nntile_grad_b =
        permute_rowmajor(nntile_grad_b_rowmajor, {K, N, B}, {2, 0, 1});

    auto a_pt = torch::from_blob(a_rowmajor.data(), {M, K, B},
        torch::TensorOptions().dtype(torch::kFloat32)).clone().set_requires_grad(true);
    auto b_pt = torch::from_blob(b_rowmajor.data(), {K, N, B},
        torch::TensorOptions().dtype(torch::kFloat32)).clone().set_requires_grad(true);
    auto a_batched = a_pt.permute({2, 0, 1});
    auto b_batched = b_pt.permute({2, 0, 1});
    auto out_pt = gemm_alpha * torch::bmm(a_batched, b_batched);

    auto grad_output = torch::full({B, M, N}, static_cast<float>(grad_fill_val),
        torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false));
    out_pt.backward(grad_output);

    compare_float_vectors(nntile_grad_a, a_pt.grad().permute({2, 0, 1}));
    compare_float_vectors(nntile_grad_b, b_pt.grad().permute({2, 0, 1}));
}

#endif // NNTILE_HAVE_TORCH
