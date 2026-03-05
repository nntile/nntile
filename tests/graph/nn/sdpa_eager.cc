/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/nn/sdpa_eager.cc
 * Tests for NNGraph sdpa_eager autograd operation.
 *
 * @version 1.1.0
 * */

#include <cmath>
#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#ifdef NNTILE_HAVE_TORCH
#   include "pytorch_helper.hh"
#   include <torch/torch.h>
#endif

#include "nntile/graph.hh"

#include "context_fixture.hh"

using namespace nntile;
using namespace nntile::graph;

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph sdpa_eager forward", "[graph][nn_graph]")
{
    NNGraph g("sdpa");

    auto* q = g.tensor({64, 8, 2, 4}, "q", DataType::FP32);
    auto* k = g.tensor({64, 8, 2, 4}, "k", DataType::FP32);
    auto* v = g.tensor({64, 8, 2, 4}, "v", DataType::FP32);

    auto* output = sdpa_eager(q, k, v, "out", nullptr, 2, 0);

    REQUIRE(output->shape() == std::vector<Index>({64, 8, 2, 4}));
    REQUIRE(output->name() == "out");

    size_t gemm_count = 0;
    size_t maxsumexp_count = 0;
    size_t softmax_count = 0;
    for(const auto& op : g.ops())
    {
        if(op->op_name() == "GEMM")
            ++gemm_count;
        if(op->op_name() == "MAXSUMEXP")
            ++maxsumexp_count;
        if(op->op_name() == "SOFTMAX_INPLACE")
            ++softmax_count;
    }
    REQUIRE(gemm_count == 2);
    REQUIRE(maxsumexp_count == 1);
    REQUIRE(softmax_count == 1);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph sdpa_eager forward with mask", "[graph][nn_graph]")
{
    NNGraph g("sdpa");

    auto* q = g.tensor({64, 8, 2, 4}, "q", DataType::FP32);
    auto* k = g.tensor({64, 8, 2, 4}, "k", DataType::FP32);
    auto* v = g.tensor({64, 8, 2, 4}, "v", DataType::FP32);
    auto* mask = g.tensor({8, 8}, "mask", DataType::BOOL);

    auto* output = sdpa_eager(q, k, v, "out", mask, 2, 0);

    REQUIRE(output->shape() == std::vector<Index>({64, 8, 2, 4}));

    size_t mask_count = 0;
    for(const auto& op : g.ops())
    {
        if(op->op_name() == "MASK_SCALAR")
            ++mask_count;
    }
    REQUIRE(mask_count == 1);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph sdpa_eager validates shape", "[graph][nn_graph]")
{
    NNGraph g("sdpa");

    auto* q = g.tensor({64, 8, 2, 4}, "q", DataType::FP32);
    auto* k = g.tensor({32, 8, 2, 4}, "k", DataType::FP32);
    auto* v = g.tensor({64, 8, 2, 4}, "v", DataType::FP32);

    REQUIRE_THROWS_AS(
        sdpa_eager(q, k, v, "out", nullptr, 2, 0),
        std::invalid_argument);
}

#ifdef NNTILE_HAVE_TORCH

using nntile::test::colmajor_to_rowmajor;
using nntile::test::permute_rowmajor;
using nntile::test::pytorch_tolerance;

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph sdpa_eager forward matches PyTorch", "[graph][nn_graph][pytorch]")
{
    const auto [head_size, n_seq, batch0, batch1, use_mask] = GENERATE(
        std::tuple{8, 6, 1, 1, false},
        std::tuple{8, 6, 2, 4, false},
        std::tuple{16, 8, 2, 4, false},
        std::tuple{32, 4, 3, 2, false},
        std::tuple{8, 6, 1, 1, true},
        std::tuple{8, 6, 2, 4, true},
        std::tuple{16, 8, 2, 4, true},
        std::tuple{32, 4, 3, 2, true});

    const std::vector<Index> shape = {head_size, n_seq, batch0, batch1};
    const std::vector<Index> mask_shape = {n_seq, n_seq};
    Index nelems = 1;
    for(auto s : shape)
        nelems *= s;

    std::vector<float> q_data(nelems);
    std::vector<float> k_data(nelems);
    std::vector<float> v_data(nelems);
    for(Index i3 = 0; i3 < shape[3]; ++i3)
        for(Index i2 = 0; i2 < shape[2]; ++i2)
            for(Index i1 = 0; i1 < shape[1]; ++i1)
                for(Index i0 = 0; i0 < shape[0]; ++i0)
                {
                    Index idx = i0 + i1 * shape[0] + i2 * shape[0] * shape[1]
                        + i3 * shape[0] * shape[1] * shape[2];
                    q_data[idx] = 0.01f * static_cast<float>((idx % 100) - 50);
                    k_data[idx] = 0.01f * static_cast<float>(((idx * 7) % 100) - 50);
                    v_data[idx] = 0.01f * static_cast<float>(((idx * 13) % 100) - 50);
                }

    std::vector<uint8_t> mask_data(n_seq * n_seq);
    if(use_mask)
    {
        for(Index i = 0; i < n_seq; ++i)
            for(Index j = 0; j < n_seq; ++j)
                mask_data[i + j * n_seq] = (i <= j) ? 1 : 0;
    }

    NNGraph g("sdpa_pytorch");
    auto* q = g.tensor(shape, "q", DataType::FP32, true);
    auto* k = g.tensor(shape, "k", DataType::FP32, true);
    auto* v = g.tensor(shape, "v", DataType::FP32, true);
    NNGraph::TensorNode* mask = nullptr;
    if(use_mask)
        mask = g.tensor(mask_shape, "mask", DataType::BOOL, false);

    auto* output = sdpa_eager(q, k, v, "out", mask, 2, 0);

    q->mark_input(true);
    k->mark_input(true);
    v->mark_input(true);
    if(mask)
        mask->mark_input(true);
    output->mark_output(true);

    TensorGraph::Runtime runtime(g.tensor_graph());
    runtime.compile();
    runtime.bind_data("q", q_data);
    runtime.bind_data("k", k_data);
    runtime.bind_data("v", v_data);
    if(mask)
        runtime.bind_data("mask", mask_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> nntile_out_colmajor =
        runtime.get_output<float>(output->name());
    std::vector<float> nntile_out_rowmajor =
        colmajor_to_rowmajor(nntile_out_colmajor, shape);
    // Batch-of-matrices: permute [h,s,b0,b1] -> [b0,b1,h,s] for comparison
    std::vector<float> nntile_out =
        permute_rowmajor(nntile_out_rowmajor, shape, {2, 3, 0, 1});

    std::vector<float> q_row = colmajor_to_rowmajor(q_data, shape);
    std::vector<float> k_row = colmajor_to_rowmajor(k_data, shape);
    std::vector<float> v_row = colmajor_to_rowmajor(v_data, shape);

    std::vector<::int64_t> shape_pt(shape.begin(), shape.end());
    auto q_pt = torch::from_blob(q_row.data(), shape_pt,
        torch::TensorOptions().dtype(torch::kFloat32)).clone();
    auto k_pt = torch::from_blob(k_row.data(), shape_pt,
        torch::TensorOptions().dtype(torch::kFloat32)).clone();
    auto v_pt = torch::from_blob(v_row.data(), shape_pt,
        torch::TensorOptions().dtype(torch::kFloat32)).clone();

    float scale = 1.0f / std::sqrt(static_cast<float>(head_size));
    auto scores = torch::einsum("hsbn,htbn->stbn", {k_pt, q_pt}) * scale;
    if(use_mask)
    {
        std::vector<uint8_t> mask_row =
            colmajor_to_rowmajor(mask_data, mask_shape);
        auto mask_pt = torch::from_blob(mask_row.data(), {n_seq, n_seq},
            torch::TensorOptions().dtype(torch::kBool)).clone();
        mask_pt = mask_pt.unsqueeze(-1).unsqueeze(-1)
            .expand({n_seq, n_seq, batch0, batch1});
        scores = torch::where(mask_pt, scores,
            torch::full_like(scores, -std::numeric_limits<float>::infinity()));
    }
    auto attn = torch::softmax(scores, 0);
    auto out_pt = torch::einsum("hsbn,stbn->htbn", {v_pt, attn});
    // Match batch-first layout for comparison
    out_pt = out_pt.permute({2, 3, 0, 1}).contiguous();

    std::vector<float> pytorch_out(out_pt.data_ptr<float>(),
                                   out_pt.data_ptr<float>() + nelems);

    REQUIRE(nntile_out.size() == pytorch_out.size());
    float max_diff = 0;
    for(size_t i = 0; i < nntile_out.size(); ++i)
        max_diff = std::max(max_diff, std::abs(nntile_out[i] - pytorch_out[i]));
    REQUIRE(max_diff < pytorch_tolerance);
}

#endif // NNTILE_HAVE_TORCH
