/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/nn_graph/sdpa_eager.cc
 * Tests for NNGraph sdpa_eager autograd operation.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <cmath>
#include <stdexcept>
#include <vector>

#ifdef NNTILE_HAVE_TORCH
#include "pytorch_helper.hh"
#include "pytorch_tile_helpers.hh"

#include <torch/torch.h>
#endif

#include "context_fixture.hh"
#include "nntile/graph.hh"

using namespace nntile;
using namespace nntile;

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph sdpa_eager forward",
    "[graph][nn_graph]")
{
    NNGraph g("sdpa");

    auto *q = g.tensor({4, 2, 8, 64}, DataType::FP32)->set_name("q");
    auto *k = g.tensor({4, 2, 8, 64}, DataType::FP32)->set_name("k");
    auto *v = g.tensor({4, 2, 8, 64}, DataType::FP32)->set_name("v");

    auto *output = sdpa_eager(q, k, v, nullptr, 2, 0)->set_name("out");

    REQUIRE(output->shape() == std::vector<Index>({4, 2, 8, 64}));
    REQUIRE(output->name() == "out");

    size_t gemm_count = 0;
    size_t maxsumexp_count = 0;
    size_t softmax_count = 0;
    for (const auto &op : g.ops())
    {
        if (op->op_name() == "GEMM")
            ++gemm_count;
        if (op->op_name() == "MAXSUMEXP")
            ++maxsumexp_count;
        if (op->op_name() == "SOFTMAX_INPLACE")
            ++softmax_count;
    }
    REQUIRE(gemm_count == 2);
    REQUIRE(maxsumexp_count == 1);
    REQUIRE(softmax_count == 1);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph sdpa_eager forward with mask",
    "[graph][nn_graph]")
{
    NNGraph g("sdpa");

    auto *q = g.tensor({4, 2, 8, 64}, DataType::FP32)->set_name("q");
    auto *k = g.tensor({4, 2, 8, 64}, DataType::FP32)->set_name("k");
    auto *v = g.tensor({4, 2, 8, 64}, DataType::FP32)->set_name("v");
    auto *mask = g.tensor({8, 8}, DataType::BOOL)->set_name("mask");

    auto *output = sdpa_eager(q, k, v, mask, 2, 0)->set_name("out");

    REQUIRE(output->shape() == std::vector<Index>({4, 2, 8, 64}));

    size_t mask_count = 0;
    for (const auto &op : g.ops())
    {
        if (op->op_name() == "MASK_SCALAR")
            ++mask_count;
    }
    REQUIRE(mask_count == 1);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph sdpa_eager validates shape",
    "[graph][nn_graph]")
{
    NNGraph g("sdpa");

    auto *q = g.tensor({4, 2, 8, 64}, DataType::FP32)->set_name("q");
    auto *k = g.tensor({4, 2, 8, 32}, DataType::FP32)->set_name("k");
    auto *v = g.tensor({4, 2, 8, 64}, DataType::FP32)->set_name("v");

    REQUIRE_THROWS_AS(
        sdpa_eager(q, k, v, nullptr, 2, 0), std::invalid_argument);
}

#ifdef NNTILE_HAVE_TORCH

using nntile::test::compare_float_vectors;
using nntile::test::nn_pytorch_tile_heterogeneous_rank4_hs_bn_b0b1;
using nntile::test::nn_pytorch_tile_mask_nn;
using nntile::test::pytorch_tolerance;

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph sdpa_eager forward and backward match PyTorch",
    "[graph][nn_graph][pytorch]")
{
    const auto [head_size, n_seq, batch0, batch1, use_mask, grad_fill_val] =
        GENERATE(std::tuple{8, 6, 1, 1, false, Scalar(1.0)},
            std::tuple{8, 6, 2, 4, false, Scalar(1.0)},
            std::tuple{16, 8, 2, 4, false, Scalar(1.0)},
            std::tuple{32, 4, 3, 2, false, Scalar(1.0)},
            std::tuple{8, 6, 1, 1, true, Scalar(1.0)},
            std::tuple{8, 6, 2, 4, true, Scalar(1.0)},
            std::tuple{16, 8, 2, 4, true, Scalar(1.0)},
            std::tuple{32, 4, 3, 2, true, Scalar(1.0)});

    const std::vector<Index> shape = {batch0, batch1, n_seq, head_size};
    const std::vector<Index> mask_shape = {n_seq, n_seq};
    Index nelems = 1;
    for (auto s : shape)
        nelems *= s;

    std::vector<float> q_data(nelems);
    std::vector<float> k_data(nelems);
    std::vector<float> v_data(nelems);
    for (Index ih = 0; ih < shape[3]; ++ih)
        for (Index is = 0; is < shape[2]; ++is)
            for (Index ib1 = 0; ib1 < shape[1]; ++ib1)
                for (Index ib0 = 0; ib0 < shape[0]; ++ib0)
                {
                    Index idx = ib0 + ib1 * shape[0] +
                                is * shape[0] * shape[1] +
                                ih * shape[0] * shape[1] * shape[2];
                    q_data[idx] = 0.01f * static_cast<float>((idx % 100) - 50);
                    k_data[idx] =
                        0.01f * static_cast<float>(((idx * 7) % 100) - 50);
                    v_data[idx] =
                        0.01f * static_cast<float>(((idx * 13) % 100) - 50);
                }

    std::vector<uint8_t> mask_data(n_seq * n_seq);
    if (use_mask)
    {
        for (Index key = 0; key < n_seq; ++key)
            for (Index query = 0; query < n_seq; ++query)
                mask_data[key + query * n_seq] = (key <= query) ? 1 : 0;
    }

    NNGraph g("sdpa_pytorch");
    auto *q = g.tensor(shape, DataType::FP32, true)->set_name("q");
    auto *k = g.tensor(shape, DataType::FP32, true)->set_name("k");
    auto *v = g.tensor(shape, DataType::FP32, true)->set_name("v");
    NNGraph::TensorNode *mask = nullptr;
    if (use_mask)
        mask = g.tensor(mask_shape, DataType::BOOL, false)->set_name("mask");

    auto *output = sdpa_eager(q, k, v, mask, 2, 0)->set_name("out");

    nn_pytorch_tile_heterogeneous_rank4_hs_bn_b0b1(q);
    nn_pytorch_tile_heterogeneous_rank4_hs_bn_b0b1(k);
    nn_pytorch_tile_heterogeneous_rank4_hs_bn_b0b1(v);
    if (mask)
        nn_pytorch_tile_mask_nn(mask);

    q->mark_input(true);
    k->mark_input(true);
    v->mark_input(true);
    if (mask)
        mask->mark_input(true);
    output->mark_output(true);

    // Build backward graph
    auto [out_grad, is_first] = g.get_or_create_grad(output, "out_grad");
    fill(grad_fill_val, out_grad);
    output->backward();

    q->grad()->mark_output(true);
    k->grad()->mark_output(true);
    v->grad()->mark_output(true);

    TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());
    Runtime runtime(tile_graph);
    runtime.compile();
    runtime.bind_data(q, q_data);
    runtime.bind_data(k, k_data);
    runtime.bind_data(v, v_data);
    if (mask)
        runtime.bind_data(mask, mask_data);
    runtime.execute();
    runtime.wait();

    // --- Forward comparison ---
    std::vector<::int64_t> shape_pt(shape.begin(), shape.end());
    std::vector<float> nntile_out =
        runtime.get_output<float>(output);

    auto q_pt = torch::from_blob(
        q_data.data(), shape_pt, torch::TensorOptions().dtype(torch::kFloat32))
                    .clone()
                    .set_requires_grad(true);
    auto k_pt = torch::from_blob(
        k_data.data(), shape_pt, torch::TensorOptions().dtype(torch::kFloat32))
                    .clone()
                    .set_requires_grad(true);
    auto v_pt = torch::from_blob(
        v_data.data(), shape_pt, torch::TensorOptions().dtype(torch::kFloat32))
                    .clone()
                    .set_requires_grad(true);

    float scale = 1.0f / std::sqrt(static_cast<float>(head_size));
    auto scores =
        torch::einsum("abcd,abed->abce", {k_pt, q_pt}) * scale;
    if (use_mask)
    {
        std::vector<uint8_t> mask_pt_buf(n_seq * n_seq);
        for (Index key = 0; key < n_seq; ++key)
            for (Index query = 0; query < n_seq; ++query)
                mask_pt_buf[key * n_seq + query] = (key <= query) ? 1 : 0;
        auto mask_pt = torch::from_blob(mask_pt_buf.data(),
            {n_seq, n_seq},
            torch::TensorOptions().dtype(torch::kBool))
                           .clone();
        mask_pt = mask_pt.unsqueeze(0).unsqueeze(0).expand(
            {batch0, batch1, n_seq, n_seq});
        scores = torch::where(mask_pt,
            scores,
            torch::full_like(scores, -std::numeric_limits<float>::infinity()));
    }
    auto attn = torch::softmax(scores, -2);
    auto out_pt = torch::einsum("abcd,abce->abed", {v_pt, attn});

    compare_float_vectors(nntile_out, out_pt);

    // --- Backward comparison ---
    auto grad_output_pt = torch::full(shape_pt,
        static_cast<float>(grad_fill_val),
        torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false));
    out_pt.backward(grad_output_pt);

    std::vector<float> nntile_grad_q =
        runtime.get_output<float>(q->grad());
    std::vector<float> nntile_grad_k =
        runtime.get_output<float>(k->grad());
    std::vector<float> nntile_grad_v =
        runtime.get_output<float>(v->grad());

    compare_float_vectors(nntile_grad_q, q_pt.grad());
    compare_float_vectors(nntile_grad_k, k_pt.grad());
    compare_float_vectors(nntile_grad_v, v_pt.grad());
}

#endif // NNTILE_HAVE_TORCH
