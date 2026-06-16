/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/nn_graph/transpose.cc
 * Test NNGraph transpose autograd operation.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>

#include "context_fixture.hh"
#include "nntile/graph.hh"
#include "nntile/tensor/ops/transpose.hh"

using namespace nntile;
namespace gt = nntile::tensor;

namespace
{

std::vector<Index> cyclic_shift(const std::vector<Index> &shape, Index ndim)
{
    const Index n = static_cast<Index>(shape.size());
    std::vector<Index> out(n);
    for (Index i = 0; i < n; ++i)
    {
        out[i] = shape[(i + ndim) % n];
    }
    return out;
}

} // anonymous namespace

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph transpose maps model storage axis to graph axis",
    "[graph][nn_graph]")
{
    NNGraph g("nn_transpose_gpt2_q");

    // After Q projection: [batch, seq, head_size, n_heads].
    const std::vector<Index> q_proj_shape = {2, 8, 16, 4};
    auto *q_proj = g.tensor(q_proj_shape, DataType::FP32)->set_name("q_proj");

    // Model code: transpose(q_proj, 1) — storage-order axis.
    auto *q = transpose(q_proj, 1)->set_name("q");

    REQUIRE(q->shape() == std::vector<Index>({4, 2, 8, 16}));
    REQUIRE(g.num_ops() == 1);
    REQUIRE(g.tensor_graph().ops()[0]->op_name() == "TRANSPOSE");

    // Raw tensor transpose with the same literal ndim would differ.
    TensorGraph tg("tensor_transpose_compare");
    auto *tg_src = tg.data(q_proj_shape)->set_name("tg_src");
    auto *tg_dst = gt::transpose(1.0, tg_src, 1);
    REQUIRE(tg_dst->shape() == cyclic_shift(q_proj_shape, 1));
    REQUIRE(tg_dst->shape() != q->shape());
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph transpose maps attn_out for output projection",
    "[graph][nn_graph]")
{
    NNGraph g("nn_transpose_gpt2_attn");

    const std::vector<Index> attn_shape = {4, 2, 8, 16};
    auto *attn_out = g.tensor(attn_shape, DataType::FP32)->set_name("attn_out");

    // Model code: transpose(attn_out, 3).
    auto *attn_t = transpose(attn_out, 3)->set_name("attn_t");

    REQUIRE(attn_t->shape() == std::vector<Index>({2, 8, 16, 4}));
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph transpose backward",
    "[graph][nn_graph]")
{
    NNGraph g("nn_transpose_backward");

    const std::vector<Index> shape = {2, 8, 16, 4};
    auto *src = g.tensor(shape, DataType::FP32, true)->set_name("src");
    auto *out = transpose(src, 1)->set_name("out");

    auto [out_grad, _] = g.get_or_create_grad(out, "out_grad");
    gt::fill(Scalar(1.0), out_grad->data());
    out->backward();

    REQUIRE(src->has_grad());
    REQUIRE(src->grad()->shape() == shape);
}

TEST_CASE("NNGraph transpose rejects invalid ndim", "[graph][nn_graph]")
{
    NNGraph g("nn_transpose_invalid");
    auto *src = g.tensor({2, 3, 4}, DataType::FP32)->set_name("src");

    REQUIRE_THROWS_AS(transpose(src, 0), std::invalid_argument);
    REQUIRE_THROWS_AS(transpose(src, 3), std::invalid_argument);
    REQUIRE_THROWS_AS(transpose(nullptr, 1), std::invalid_argument);
}
