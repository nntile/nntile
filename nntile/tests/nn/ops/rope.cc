/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/nn_graph/rope.cc
 * Test NNGraph rope autograd operation.
 *
 * @version 1.1.0
 * */

#include "context_fixture.hh"
#include "nntile/graph.hh"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

using namespace nntile;
using namespace nntile;

namespace
{
constexpr float float_tolerance = 1e-5f;

void set_rope_heterogeneous_tiling(NNGraph::TensorNode *sin,
    NNGraph::TensorNode *cos,
    NNGraph::TensorNode *src)
{
    const Index rope_axis = sin->ndim() - 1;
    for (Index d = 0; d < sin->ndim(); ++d)
    {
        const Index Ls = sin->shape()[static_cast<size_t>(d)];
        std::vector<Index> sin_seg;
        if (Ls >= 4)
        {
            sin_seg = {1, Ls - 1};
        }
        else if (Ls == 3)
        {
            sin_seg = {1, 2};
        }
        else if (Ls == 2)
        {
            sin_seg = {1, 1};
        }
        else
        {
            sin_seg = {Ls};
        }
        sin->data()->axis(d)->set_tiling(sin_seg);
        cos->data()->axis(d)->set_tiling(sin_seg);
        if (d == rope_axis)
        {
            std::vector<Index> src_seg;
            src_seg.reserve(sin_seg.size());
            for (Index v : sin_seg)
            {
                src_seg.push_back(2 * v);
            }
            src->data()->axis(rope_axis)->set_tiling(std::move(src_seg));
        }
        else
        {
            src->data()->axis(d)->set_tiling(sin_seg);
        }
    }
}
} // namespace

// RoPE pairs live on the last sin axis; src doubles that extent.
static std::vector<Index> make_src_shape(const std::vector<Index> &sin_shape)
{
    std::vector<Index> src_shape = sin_shape;
    src_shape.back() *= 2;
    return src_shape;
}

static Index rope_n_for_graph(
    const std::vector<Index> &src_shape,
    const std::vector<Index> &sin_shape)
{
    if (src_shape.size() > sin_shape.size())
    {
        Index n = 1;
        for (Index d = static_cast<Index>(sin_shape.size());
            d < static_cast<Index>(src_shape.size()); ++d)
        {
            n *= src_shape[static_cast<size_t>(d)];
        }
        return n;
    }
    Index n = 1;
    const Index rope_axis = static_cast<Index>(sin_shape.size()) - 1;
    for (Index d = 0; d < rope_axis; ++d)
    {
        n *= src_shape[static_cast<size_t>(d)];
    }
    return n;
}

static Index rope_m_pairs(const std::vector<Index> &src_shape,
    const std::vector<Index> &sin_shape)
{
    const Index rope_axis = static_cast<Index>(sin_shape.size()) - 1;
    return src_shape[static_cast<size_t>(rope_axis)] / 2;
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph rope structure",
    "[graph][nn_graph]")
{
    const auto sin_shape = GENERATE(std::vector<Index>{2});
    const auto src_shape = make_src_shape(sin_shape);

    NNGraph g("rope_structure");
    auto *sin = g.tensor(sin_shape, DataType::FP32, false)->set_name("sin");
    auto *cos = g.tensor(sin_shape, DataType::FP32, false)->set_name("cos");
    auto *x = g.tensor(src_shape, DataType::FP32)->set_name("x");
    auto *y = rope(sin, cos, x)->set_name("y");

    REQUIRE(y != nullptr);
    REQUIRE(y->has_producer());
    REQUIRE(y->shape() == src_shape);
    REQUIRE(g.num_ops() >= 1);
}

TEST_CASE_METHOD(
    nntile::test::ContextFixture, "NNGraph rope backward", "[graph][nn_graph]")
{
    const auto [sin_shape, grad_fill_val] =
        GENERATE(std::tuple{std::vector<Index>{2}, Scalar(1.0)},
            std::tuple{std::vector<Index>{2}, Scalar(-1.0)});
    const auto src_shape = make_src_shape(sin_shape);

    NNGraph g("rope_backward");
    auto *sin = g.tensor(sin_shape, DataType::FP32, false)->set_name("sin");
    auto *cos = g.tensor(sin_shape, DataType::FP32, false)->set_name("cos");
    auto *x = g.tensor(src_shape, DataType::FP32)->set_name("x");
    auto *y = rope(sin, cos, x)->set_name("y");

    auto [y_grad, _] = g.get_or_create_grad(y, "y_grad");
    fill(grad_fill_val, y_grad);
    y->backward();

    REQUIRE(x->has_grad());
    REQUIRE(x->grad()->shape() == src_shape);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph rope forward and backward",
    "[graph][nn_graph]")
{
    const auto [sin_shape, grad_fill_val] =
        GENERATE(std::tuple{std::vector<Index>{2}, Scalar(1.0)},
            std::tuple{std::vector<Index>{2}, Scalar(1.0)},
            std::tuple{std::vector<Index>{2}, Scalar(-1.0)});
    const auto src_shape = make_src_shape(sin_shape);

    NNGraph g("rope");
    auto *sin = g.tensor(sin_shape, DataType::FP32, false)->set_name("sin");
    auto *cos = g.tensor(sin_shape, DataType::FP32, false)->set_name("cos");
    auto *x = g.tensor(src_shape, DataType::FP32, true)->set_name("x");
    auto *y = rope(sin, cos, x)->set_name("y");

    REQUIRE(y != nullptr);
    REQUIRE(y->has_producer());
    REQUIRE(y->shape() == x->shape());

    auto [y_grad, _] = g.get_or_create_grad(y, "y_grad");
    fill(grad_fill_val, y_grad);
    y->backward();

    REQUIRE(x->has_grad());
    REQUIRE(x->grad()->shape() == x->shape());
}

static void rope_forward_ref(const float *sin_data,
    const float *cos_data,
    const float *src_data,
    float *dst_data,
    Index m_pairs,
    Index n)
{
    for (Index j = 0; j < n; ++j)
        for (Index i = 0; i < m_pairs; ++i)
        {
            const Index si = i * n + j;
            const Index l0 = 2 * i * n + j;
            const Index l1 = l0 + n;
            float c = cos_data[si];
            float s = sin_data[si];
            float a = src_data[l0];
            float b = src_data[l1];
            dst_data[l0] = c * a - s * b;
            dst_data[l1] = s * a + c * b;
        }
}

static void rope_backward_ref(const float *sin_data,
    const float *cos_data,
    const float *dy_data,
    float *dx_data,
    Index m_pairs,
    Index n)
{
    for (Index j = 0; j < n; ++j)
        for (Index i = 0; i < m_pairs; ++i)
        {
            const Index si = i * n + j;
            const Index l0 = 2 * i * n + j;
            const Index l1 = l0 + n;
            float c = cos_data[si];
            float s = sin_data[si];
            float a = dy_data[l0];
            float b = dy_data[l1];
            dx_data[l0] = c * a + s * b;
            dx_data[l1] = c * b - s * a;
        }
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph rope forward matches reference",
    "[graph][nn_graph]")
{
    const auto sin_shape = GENERATE(std::vector<Index>{2});
    const auto src_shape = make_src_shape(sin_shape);

    Index sin_nelems = 1;
    for (auto s : sin_shape)
        sin_nelems *= s;
    const Index src_nelems = std::accumulate(
        src_shape.begin(), src_shape.end(), Index(1), std::multiplies<>());

    std::vector<float> sin_data(sin_nelems);
    std::vector<float> cos_data(sin_nelems);
    std::vector<float> src_data(src_nelems);
    for (Index i = 0; i < sin_nelems; ++i)
    {
        sin_data[i] = 0.1f * static_cast<float>((i % 10));
        cos_data[i] = 0.1f * static_cast<float>(((i + 1) % 10));
    }
    for (Index i = 0; i < src_nelems; ++i)
        src_data[i] = 0.15f * static_cast<float>(i - src_nelems / 2);

    NNGraph g("rope_ref");
    auto *sin = g.tensor(sin_shape, DataType::FP32, false)->set_name("sin");
    auto *cos = g.tensor(sin_shape, DataType::FP32, false)->set_name("cos");
    auto *x = g.tensor(src_shape, DataType::FP32, true)->set_name("x");
    auto *y = rope(sin, cos, x)->set_name("y");

    sin->mark_input(true);
    cos->mark_input(true);
    x->mark_input(true);
    y->mark_output(true);

    set_rope_heterogeneous_tiling(sin, cos, x);

    TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());
    Runtime runtime(tile_graph);
    runtime.compile();
    runtime.bind_data(sin, sin_data);
    runtime.bind_data(cos, cos_data);
    runtime.bind_data(x, src_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> nntile_out = runtime.get_output<float>(y);

    std::vector<float> ref_out(src_nelems);
    const Index m_pairs = rope_m_pairs(src_shape, sin_shape);
    const Index n = rope_n_for_graph(src_shape, sin_shape);
    rope_forward_ref(sin_data.data(),
        cos_data.data(),
        src_data.data(),
        ref_out.data(),
        m_pairs,
        n);

    REQUIRE(nntile_out.size() == ref_out.size());
    for (size_t i = 0; i < nntile_out.size(); ++i)
        REQUIRE(std::abs(nntile_out[i] - ref_out[i]) < float_tolerance);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph rope backward matches reference",
    "[graph][nn_graph]")
{
    const auto [sin_shape, grad_fill_val] =
        GENERATE(std::tuple{std::vector<Index>{2}, Scalar(1.0)},
            std::tuple{std::vector<Index>{2}, Scalar(-1.0)});
    const auto src_shape = make_src_shape(sin_shape);

    Index sin_nelems = 1;
    for (auto s : sin_shape)
        sin_nelems *= s;
    const Index src_nelems = std::accumulate(
        src_shape.begin(), src_shape.end(), Index(1), std::multiplies<>());

    std::vector<float> sin_data(sin_nelems);
    std::vector<float> cos_data(sin_nelems);
    std::vector<float> src_data(src_nelems);
    for (Index i = 0; i < sin_nelems; ++i)
    {
        sin_data[i] = 0.1f * static_cast<float>((i % 10));
        cos_data[i] = 0.1f * static_cast<float>(((i + 1) % 10));
    }
    for (Index i = 0; i < src_nelems; ++i)
        src_data[i] = 0.2f * static_cast<float>(i - src_nelems / 3);

    NNGraph g("rope_bwd_ref");
    auto *sin = g.tensor(sin_shape, DataType::FP32, false)->set_name("sin");
    auto *cos = g.tensor(sin_shape, DataType::FP32, false)->set_name("cos");
    auto *x = g.tensor(src_shape, DataType::FP32, true)->set_name("x");
    auto *y = rope(sin, cos, x)->set_name("y");

    sin->mark_input(true);
    cos->mark_input(true);
    x->mark_input(true);

    auto [y_grad, _] = g.get_or_create_grad(y, "y_grad");
    fill(grad_fill_val, y_grad);
    y->backward();

    x->grad()->mark_output(true);

    set_rope_heterogeneous_tiling(sin, cos, x);

    TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());
    Runtime runtime(tile_graph);
    runtime.compile();
    runtime.bind_data(sin, sin_data);
    runtime.bind_data(cos, cos_data);
    runtime.bind_data(x, src_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> nntile_grad_x = runtime.get_output<float>(x->grad());

    const Index m_pairs = rope_m_pairs(src_shape, sin_shape);
    const Index n = rope_n_for_graph(src_shape, sin_shape);
    std::vector<float> grad_y_data(
        src_nelems, static_cast<float>(grad_fill_val));
    std::vector<float> ref_grad_x(src_nelems);
    rope_backward_ref(sin_data.data(),
        cos_data.data(),
        grad_y_data.data(),
        ref_grad_x.data(),
        m_pairs,
        n);

    REQUIRE(nntile_grad_x.size() == ref_grad_x.size());
    for (size_t i = 0; i < nntile_grad_x.size(); ++i)
        REQUIRE(std::abs(nntile_grad_x[i] - ref_grad_x[i]) < float_tolerance);
}
