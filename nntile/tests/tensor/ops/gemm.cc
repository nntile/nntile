/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/tensor_graph/gemm.cc
 * Test TensorGraph gemm operation.
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/ops/gemm.hh"

#include "context_fixture.hh"
#include "nntile/constants.hh"
#include "nntile/tensor.hh"
#include "nntile/tensor/axis_descriptor.hh"
#include "nntile/tile.hh"
#include "nntile/tensor/ops/clear.hh"
#include "nntile/tensor/ops/gemm.hh"
#include "nntile/tensor.hh"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <numeric>

using namespace nntile;
using namespace nntile;
namespace gt = nntile::tensor;

namespace
{

constexpr Scalar alpha_one = 1.0;
constexpr bool trans_a = false;
constexpr bool trans_b = false;
constexpr Index ndim = 1;
constexpr Index batch_ndim = 0;
constexpr float tolerance = 1e-4f;
constexpr int distr_rank_single = 0;

} 

TEST_CASE("TensorGraph gemm structure", "[graph][tensor]")
{
    TensorGraph graph("test");

    auto *a = graph.data({5, 4})->set_name("a");
    auto *b = graph.data({5, 6})->set_name("b");
    auto *c = gt::gemm(a, b, alpha_one, trans_a, trans_b, ndim, batch_ndim);

    REQUIRE(graph.num_data() == 3);
    REQUIRE(graph.num_ops() == 1);
    REQUIRE(c->shape().size() == 2);
    REQUIRE(c->shape()[0] == 6);
    REQUIRE(c->shape()[1] == 4);

    const auto &ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "GEMM");
    REQUIRE(ops[0]->inputs().size() == 3);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == c);
}

TEST_CASE("TensorGraph gemm rejects null", "[graph][tensor]")
{
    TensorGraph graph("test");
    auto *a = graph.data({5, 4})->set_name("a");
    auto *b = graph.data({5, 6})->set_name("b");

    REQUIRE_THROWS_AS(
        gt::gemm(nullptr, b, alpha_one, trans_a, trans_b, ndim, batch_ndim),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        gt::gemm(a, nullptr, alpha_one, trans_a, trans_b, ndim, batch_ndim),
        std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph gemm tiled matches untiled",
    "[graph][tensor]")
{
    const auto [M, K, N, alpha] =
        GENERATE(std::tuple{Index(4), Index(5), Index(6), 1.0},
            std::tuple{Index(2), Index(2), Index(2), 1.0});

    using Y = nntile::fp32_t::repr_t;
    std::vector<Index> a_shape = {K, M};
    std::vector<Index> b_shape = {K, N};

    const Index a_nelems = M * K;
    const Index b_nelems = K * N;

    std::vector<float> a_data(a_nelems);
    std::vector<float> b_data(b_nelems);
    for (Index k = 0; k < K; ++k)
    {
        for (Index m = 0; m < M; ++m)
        {
            const Index i = k * M + m;
            a_data[static_cast<size_t>(i)] =
                static_cast<float>(Y(i % 10)) * 0.1f;
        }
    }
    for (Index k = 0; k < K; ++k)
    {
        for (Index n = 0; n < N; ++n)
        {
            const Index i = k * N + n;
            b_data[static_cast<size_t>(i)] =
                static_cast<float>(Y(i % 7)) * 0.1f;
        }
    }

    // --- Untiled run ---
    std::vector<float> untiled_result;
    {
        TensorGraph graph("gemm_untiled");
        auto *a_node = graph.data(a_shape, DataType::FP32)->set_name("a");
        auto *b_node = graph.data(b_shape, DataType::FP32)->set_name("b");
        a_node->mark_input(true);
        b_node->mark_input(true);

        auto *c_node = gt::gemm(
            a_node, b_node, alpha, trans_a, trans_b, ndim, batch_ndim);
        c_node->mark_output(true);

        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);

        Runtime runtime(tile_graph);
        runtime.compile();

        runtime.bind_data(a_node, a_data);
        runtime.bind_data(b_node, b_data);
        runtime.execute();
        runtime.wait();

        untiled_result = runtime.get_output<float>(c_node);
    }

    // --- Tiled run ---
    std::vector<float> tiled_result;
    {
        TensorGraph graph("gemm_tiled");
        auto *a_node = graph.data(a_shape, DataType::FP32)->set_name("a");
        auto *b_node = graph.data(b_shape, DataType::FP32)->set_name("b");
        a_node->mark_input(true);
        b_node->mark_input(true);

        auto *c_node = gt::gemm(
            a_node, b_node, alpha, trans_a, trans_b, ndim, batch_ndim);
        c_node->mark_output(true);
        for (auto *ag : graph.axis_groups())
        {
            ag->set_tiling((ag->extent + 1) / 2);
        }

        TileGraph tile_graph = TileGraph::from_tensor_graph(graph);

        Runtime runtime(tile_graph);
        runtime.compile();

        runtime.bind_data(a_node, a_data);
        runtime.bind_data(b_node, b_data);
        runtime.execute();
        runtime.wait();

        tiled_result = runtime.get_output<float>(c_node);
    }

    // --- Compare ---
    constexpr float tol = 1e-4f;
    REQUIRE(tiled_result.size() == untiled_result.size());
    for (size_t i = 0; i < tiled_result.size(); ++i)
    {
        REQUIRE(std::abs(untiled_result[i] - tiled_result[i]) < tol);
    }
}

namespace
{

std::vector<float> ref_gemm_false_true_ndim1(const std::vector<float> &a,
    const std::vector<Index> &a_shape,
    const std::vector<float> &b,
    const std::vector<Index> &b_shape)
{
    const Index a0 = a_shape[0];
    const Index a1 = a_shape[1];
    const Index a2 = a_shape[2];
    const Index b0 = b_shape[0];
    const Index b1 = b_shape[1];
    const Index b2 = b_shape[2];
    std::vector<float> out(static_cast<size_t>(b0 * b1 * a1 * a2), 0.f);
    for (Index bi = 0; bi < b0; ++bi)
    {
        for (Index bj = 0; bj < b1; ++bj)
        {
            for (Index ai = 0; ai < a1; ++ai)
            {
                for (Index aj = 0; aj < a2; ++aj)
                {
                    float sum = 0.f;
                    for (Index k = 0; k < a0; ++k)
                    {
                        const float av = a[static_cast<size_t>(
                            (k * a1 + ai) * a2 + aj)];
                        const float bv = b[static_cast<size_t>(
                            (bi * b1 + bj) * b2 + k)];
                        sum += av * bv;
                    }
                    out[static_cast<size_t>(
                        ((bi * b1 + bj) * a1 + ai) * a2 + aj)] = sum;
                }
            }
        }
    }
    return out;
}

std::vector<float> ref_gemm_true_true_ndim1(const std::vector<float> &a,
    const std::vector<Index> &a_shape,
    const std::vector<float> &b,
    const std::vector<Index> &b_shape)
{
    const Index a0 = a_shape[0];
    const Index a1 = a_shape[1];
    const Index b0 = b_shape[0];
    const Index b1 = b_shape[1];
    std::vector<float> out(static_cast<size_t>(b0 * a0), 0.f);
    for (Index bi = 0; bi < b0; ++bi)
    {
        for (Index ao = 0; ao < a0; ++ao)
        {
            float sum = 0.f;
            for (Index k = 0; k < a1; ++k)
            {
                sum += a[static_cast<size_t>(ao * a1 + k)] *
                       b[static_cast<size_t>(bi * b1 + k)];
            }
            out[static_cast<size_t>(bi * a0 + ao)] = sum;
        }
    }
    return out;
}

std::vector<float> ref_gemm_false_true_ndim2(const std::vector<float> &a,
    const std::vector<Index> &a_shape,
    const std::vector<float> &b,
    const std::vector<Index> &b_shape)
{
    const Index a0 = a_shape[0];
    const Index a1 = a_shape[1];
    const Index a2 = a_shape[2];
    const Index b0 = b_shape[0];
    const Index b1 = b_shape[1];
    const Index b2 = b_shape[2];
    const Index b3 = b_shape[3];
    std::vector<float> out(static_cast<size_t>(b0 * b1 * a2), 0.f);
    for (Index bi = 0; bi < b0; ++bi)
    {
        for (Index bj = 0; bj < b1; ++bj)
        {
            for (Index ao = 0; ao < a2; ++ao)
            {
                float sum = 0.f;
                for (Index ak = 0; ak < a0; ++ak)
                {
                    for (Index al = 0; al < a1; ++al)
                    {
                        const float av = a[static_cast<size_t>(
                            (ak * a1 + al) * a2 + ao)];
                        const float bv = b[static_cast<size_t>(
                            (((bi * b1 + bj) * b2 + ak) * b3 + al))];
                        sum += av * bv;
                    }
                }
                out[static_cast<size_t>((bi * b1 + bj) * a2 + ao)] = sum;
            }
        }
    }
    return out;
}

float frob_rel(const std::vector<float> &x, const std::vector<float> &y)
{
    double sq_diff = 0.0;
    double sq_x = 0.0;
    double sq_y = 0.0;
    for (size_t i = 0; i < x.size(); ++i)
    {
        const double d = static_cast<double>(x[i]) - static_cast<double>(y[i]);
        sq_diff += d * d;
        sq_x += static_cast<double>(x[i]) * static_cast<double>(x[i]);
        sq_y += static_cast<double>(y[i]) * static_cast<double>(y[i]);
    }
    const double scale =
        std::max(std::sqrt(sq_x), std::max(std::sqrt(sq_y), 1e-7));
    return static_cast<float>(std::sqrt(sq_diff) / scale);
}

} // namespace

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph gemm linear pattern matches reference",
    "[graph][tensor]")
{
    constexpr Index batch = 2;
    constexpr Index in_dim = 3;
    constexpr Index out_dim = 4;

    std::vector<Index> w_shape = {out_dim, in_dim};
    std::vector<Index> x_shape = {batch, in_dim};
    std::vector<float> w_data(out_dim * in_dim);
    std::vector<float> x_data(batch * in_dim);
    for (size_t i = 0; i < w_data.size(); ++i)
        w_data[i] = 0.1f * static_cast<float>(i + 1);
    for (size_t i = 0; i < x_data.size(); ++i)
        x_data[i] = 0.15f * static_cast<float>(i + 2);

    const std::vector<float> ref =
        ref_gemm_true_true_ndim1(w_data, w_shape, x_data, x_shape);

    TensorGraph graph("gemm_linear");
    auto *w = graph.data(w_shape, DataType::FP32)->set_name("w");
    auto *x = graph.data(x_shape, DataType::FP32)->set_name("x");
    auto *y = gt::gemm(w, x, alpha_one, true, true, ndim, batch_ndim);
    w->mark_input(true);
    x->mark_input(true);
    y->mark_output(true);

    TileGraph tile_graph = TileGraph::from_tensor_graph(graph);
    Runtime runtime(tile_graph);
    runtime.compile();
    runtime.bind_data(w, w_data);
    runtime.bind_data(x, x_data);
    runtime.execute();
    runtime.wait();

    REQUIRE(frob_rel(runtime.get_output<float>(y), ref) < 1e-6f);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph gemm attention Q pattern matches reference",
    "[graph][tensor]")
{
    constexpr Index batch = 2;
    constexpr Index seq = 8;
    constexpr Index hidden = 64;
    constexpr Index head_size = 16;
    constexpr Index n_heads = 4;

    std::vector<Index> w_shape = {hidden, head_size, n_heads};
    std::vector<Index> x_shape = {batch, seq, hidden};
    std::vector<float> w_data(hidden * head_size * n_heads);
    std::vector<float> x_data(batch * seq * hidden);
    for (size_t i = 0; i < w_data.size(); ++i)
        w_data[i] = 0.1f * static_cast<float>(i + 1);
    for (size_t i = 0; i < x_data.size(); ++i)
        x_data[i] = 0.15f * static_cast<float>(i + 2);

    const std::vector<float> ref =
        ref_gemm_false_true_ndim1(w_data, w_shape, x_data, x_shape);

    TensorGraph graph("gemm_attn_q");
    auto *w = graph.data(w_shape, DataType::FP32)->set_name("w");
    auto *x = graph.data(x_shape, DataType::FP32)->set_name("x");
    auto *y = gt::gemm(w, x, alpha_one, false, true, ndim, batch_ndim);
    w->mark_input(true);
    x->mark_input(true);
    y->mark_output(true);

    TileGraph tile_graph = TileGraph::from_tensor_graph(graph);
    Runtime runtime(tile_graph);
    runtime.compile();
    runtime.bind_data(w, w_data);
    runtime.bind_data(x, x_data);
    runtime.execute();
    runtime.wait();

    REQUIRE(frob_rel(runtime.get_output<float>(y), ref) < 1e-6f);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph gemm attention output pattern matches reference",
    "[graph][tensor]")
{
    constexpr Index batch = 2;
    constexpr Index seq = 8;
    constexpr Index hidden = 64;
    constexpr Index head_size = 16;
    constexpr Index n_heads = 4;

    std::vector<Index> w_shape = {head_size, n_heads, hidden};
    std::vector<Index> attn_shape = {batch, seq, head_size, n_heads};
    std::vector<float> w_data(head_size * n_heads * hidden);
    std::vector<float> attn_data(batch * seq * head_size * n_heads);
    for (size_t i = 0; i < w_data.size(); ++i)
        w_data[i] = 0.1f * static_cast<float>(i + 1);
    for (size_t i = 0; i < attn_data.size(); ++i)
        attn_data[i] = 0.15f * static_cast<float>(i + 2);

    const std::vector<float> ref =
        ref_gemm_false_true_ndim2(w_data, w_shape, attn_data, attn_shape);

    TensorGraph graph("gemm_attn_out");
    auto *w = graph.data(w_shape, DataType::FP32)->set_name("w");
    auto *attn = graph.data(attn_shape, DataType::FP32)->set_name("attn");
    auto *y = gt::gemm(w, attn, alpha_one, false, true, 2, batch_ndim);
    w->mark_input(true);
    attn->mark_input(true);
    y->mark_output(true);

    TileGraph tile_graph = TileGraph::from_tensor_graph(graph);
    Runtime runtime(tile_graph);
    runtime.compile();
    runtime.bind_data(w, w_data);
    runtime.bind_data(attn, attn_data);
    runtime.execute();
    runtime.wait();

    REQUIRE(frob_rel(runtime.get_output<float>(y), ref) < 1e-6f);
}
