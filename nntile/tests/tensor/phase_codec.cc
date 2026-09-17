/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/tensor/phase_codec.cc
 * Tests for TensorGraph Flush PhaseIR encode/decode.
 *
 * @version 1.1.0
 * */

#include <nntile/defs.h>
#include <nntile/tensor.hh>
#include <nntile/tensor/ops/adam_step.hh>
#include <nntile/tensor/ops/add.hh>
#include <nntile/tensor/ops/conv2d_inplace.hh>
#include <nntile/tensor/ops/copy_intersection.hh>
#include <nntile/tensor/ops/embedding.hh>
#include <nntile/tensor/ops/gather.hh>
#include <nntile/tensor/ops/gelu.hh>
#include <nntile/tensor/ops/gemm.hh>
#include <nntile/tensor/ops/randn.hh>
#include <nntile/tensor/ops/scatter.hh>
#include <nntile/tensor/ops/unregister.hh>
#include <nntile/tensor/phase_codec.hh>
#ifdef NNTILE_TORCH_NATIVE_OPS
#include <nntile/tensor/ops/torch_dispatch.hh>
#endif

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <string>

using namespace nntile;
namespace gt = nntile::tensor;

namespace
{

nlohmann::json unknown_op_blob(char const *name)
{
    nlohmann::json blob;
    blob["nodes"] = nlohmann::json::array();
    nlohmann::json ops = nlohmann::json::array();
    ops.push_back(
        {
            {"op_name", name},
            {"inputs", nlohmann::json::array()},
            {"outputs", nlohmann::json::array()},
        });
    blob["ops"] = std::move(ops);
    return blob;
}

nlohmann::json torch_unary_fixture()
{
    return nlohmann::json::parse(R"({
        "nodes": [
            {"id": 1, "shape": [2], "dtype": "float32", "name": "x"},
            {"id": 2, "shape": [2], "dtype": "float32", "name": "y"}
        ],
        "ops": [
            {
                "op_name": "TORCH_UNARY",
                "inputs": [1],
                "outputs": [2],
                "attrs": {"kind": 10}
            }
        ]
    })");
}

} // namespace

TEST_CASE(
    "decode_phase allowlists TORCH_UNARY fixture",
    "[graph][tensor][codec]")
{
    auto const blob = torch_unary_fixture();
    auto const decoded = gt::decode_phase(blob);
    REQUIRE(decoded.second.size() == 1);
    REQUIRE(decoded.second.at(0).at("op_name") == "TORCH_UNARY");
    auto const encoded = gt::encode_phase(decoded.first, decoded.second);
    REQUIRE(encoded.at("ops").at(0).at("op_name") == "TORCH_UNARY");
    REQUIRE(encoded.at("nodes").size() == 2);
}

TEST_CASE("decode_phase fails closed on unknown op", "[graph][tensor][codec]")
{
    REQUIRE_THROWS_WITH(
        gt::decode_phase(unknown_op_blob("CONV3D")),
        Catch::Matchers::ContainsSubstring("UnknownOp"));
}

TEST_CASE("encode_phase fails closed on unknown op", "[graph][tensor][codec]")
{
    nlohmann::json ops = nlohmann::json::array();
    ops.push_back(
        {
            {"op_name", "FFT"},
            {"inputs", nlohmann::json::array()},
            {"outputs", nlohmann::json::array()},
        });
    REQUIRE_THROWS_WITH(
        gt::encode_phase(nlohmann::json::array(), ops),
        Catch::Matchers::ContainsSubstring("UnknownOp"));
}

TEST_CASE("v1 allowlist names round-trip", "[graph][tensor][codec]")
{
    nlohmann::json ops = nlohmann::json::array();
    for (char const *name : gt::kV1PhaseOps)
    {
        REQUIRE(gt::is_v1_phase_op(name));
        ops.push_back(
            {
                {"op_name", name},
                {"inputs", nlohmann::json::array()},
                {"outputs", nlohmann::json::array()},
                {"attrs", nlohmann::json::object()},
            });
    }
    auto const blob = gt::encode_phase(nlohmann::json::array(), ops);
    auto const decoded = gt::decode_phase(blob);
    constexpr std::size_t n_ops = gt::kV1PhaseOpCount;
    REQUIRE(decoded.second.size() == n_ops);
    REQUIRE(n_ops == 80);
    REQUIRE(decoded.second.at(0).at("op_name") == "TORCH_UNARY");
    REQUIRE(decoded.second.at(1).at("op_name") == "TORCH_BINARY");
    REQUIRE(decoded.second.at(2).at("op_name") == "TORCH_TERNARY");
    REQUIRE(decoded.second.at(3).at("op_name") == "GATHER");
    REQUIRE(decoded.second.at(4).at("op_name") == "SCATTER");
    REQUIRE(decoded.second.at(5).at("op_name") == "UNREGISTER");
    REQUIRE(decoded.second.at(25).at("op_name") == "GELU");
    REQUIRE(gt::is_v1_phase_op("GELU"));
    REQUIRE(gt::is_v1_phase_op("FLASH_SDPA_FWD_CUDNN"));
    REQUIRE_FALSE(gt::is_v1_phase_op("FFT"));
}

TEST_CASE("empty PhaseIR decodes", "[graph][tensor][codec]")
{
    auto const decoded = gt::decode_phase(nlohmann::json::object());
    REQUIRE(decoded.first.empty());
    REQUIRE(decoded.second.empty());
    auto const encoded = gt::encode_phase(
        nlohmann::json(),
        nlohmann::json());
    REQUIRE(encoded.at("nodes").empty());
    REQUIRE(encoded.at("ops").empty());
}

TEST_CASE(
    "encode_phase TensorGraph SCATTER GATHER",
    "[graph][tensor][codec]")
{
    TensorGraph graph("codec_move");
    TensorRef src = graph.data({4});
    src->set_name("src");
    TensorRef dst = graph.data({4});
    dst->set_name("dst");
    gt::scatter(src, dst);
    TensorRef gathered = TensorRef::adopt(gt::gather(dst));

    auto const blob = gt::encode_phase(graph);
    REQUIRE(blob.at("ops").size() == 2);
    REQUIRE(blob.at("ops").at(0).at("op_name") == "SCATTER");
    REQUIRE(blob.at("ops").at(1).at("op_name") == "GATHER");
    REQUIRE(gathered->id() == blob.at("ops").at(1).at("outputs").at(0));
}

TEST_CASE(
    "encode_phase TensorGraph UNREGISTER",
    "[graph][tensor][codec]")
{
    TensorGraph graph("codec_unreg");
    TensorRef x = graph.data({2, 2});
    x->set_name("x");
    gt::unregister(x);

    auto const blob = gt::encode_phase(graph);
    REQUIRE(blob.at("ops").size() == 1);
    REQUIRE(blob.at("ops").at(0).at("op_name") == "UNREGISTER");
    REQUIRE(blob.at("ops").at(0).at("inputs").size() == 1);
    REQUIRE(blob.at("ops").at(0).at("outputs").empty());
    REQUIRE(blob.at("ops").at(0).at("inputs").at(0) == x->id());

    auto const decoded = gt::decode_phase(blob);
    REQUIRE(decoded.second.at(0).at("op_name") == "UNREGISTER");
    REQUIRE(decoded.second.at(0).at("outputs").empty());
}

TEST_CASE(
    "encode_phase TensorRef last-drop UNREGISTER",
    "[graph][tensor][codec]")
{
    TensorGraph graph("codec_drop");
    {
        TensorRef x = graph.data({2, 2});
        x->set_name("x");
    }
    auto const blob = gt::encode_phase(graph);
    REQUIRE(blob.at("ops").size() == 1);
    REQUIRE(blob.at("ops").at(0).at("op_name") == "UNREGISTER");
    REQUIRE(blob.at("ops").at(0).at("outputs").empty());
}

TEST_CASE(
    "encode_phase TensorGraph GELU is allowlisted",
    "[graph][tensor][codec]")
{
    TensorGraph graph("codec_gelu");
    TensorRef x = graph.data({2, 2});
    TensorRef y = TensorRef::adopt(gt::gelu(x));
    auto const blob = gt::encode_phase(graph);
    REQUIRE(blob.at("ops").size() == 1);
    REQUIRE(blob.at("ops").at(0).at("op_name") == "GELU");
    REQUIRE(blob.at("ops").at(0).at("attrs").empty());
    REQUIRE(y->id() == blob.at("ops").at(0).at("outputs").at(0));
}

TEST_CASE("encode_phase uses unsealed snapshot", "[graph][tensor][codec]")
{
    TensorGraph graph("codec_phase");
    TensorRef src = graph.data({4});
    TensorRef dst = graph.data({4});
    gt::scatter(src, dst);
    // Keep TensorRefs so last-drop UNREGISTER is not mixed into the
    // sealed SCATTER phase.
    auto const sealed = graph.seal_phase();
    REQUIRE_FALSE(sealed.empty());

    auto const pending = gt::encode_phase(graph);
    REQUIRE(pending.at("ops").empty());

    auto const encoded = gt::encode_phase(graph, sealed);
    REQUIRE(encoded.at("ops").size() == 1);
    REQUIRE(encoded.at("ops").at(0).at("op_name") == "SCATTER");
}

#ifdef NNTILE_TORCH_NATIVE_OPS
TEST_CASE(
    "encode_phase TensorGraph torch Relu is TORCH_UNARY",
    "[graph][tensor][codec]")
{
    TensorGraph graph("codec_torch_relu");
    TensorRef x = graph.data({2, 2});
    TensorRef y = TensorRef::adopt(
        gt::torch_unary(starpu::TorchKind::Relu, x, {2, 2}));

    auto const blob = gt::encode_phase(graph);
    REQUIRE(blob.at("ops").size() == 1);
    REQUIRE(blob.at("ops").at(0).at("op_name") == "TORCH_UNARY");
    REQUIRE(blob.at("ops").at(0).at("attrs").at("kind").get<std::int32_t>() ==
        static_cast<std::int32_t>(starpu::TorchKind::Relu));
    REQUIRE(y->id() == blob.at("ops").at(0).at("outputs").at(0));
}

TEST_CASE(
    "encode_phase TensorGraph torch Add is TORCH_BINARY",
    "[graph][tensor][codec]")
{
    TensorGraph graph("codec_torch_add");
    TensorRef a = graph.data({2, 2});
    TensorRef b = graph.data({2, 2});
    TensorRef torch_out = TensorRef::adopt(
        gt::torch_binary(starpu::TorchKind::Add, a, b, {2, 2}));

    auto const blob = gt::encode_phase(graph);
    REQUIRE(blob.at("ops").size() == 1);
    REQUIRE(blob.at("ops").at(0).at("op_name") == "TORCH_BINARY");
    REQUIRE(blob.at("ops").at(0).at("attrs").at("kind").get<std::int32_t>() ==
        static_cast<std::int32_t>(starpu::TorchKind::Add));
    REQUIRE(torch_out->id() == blob.at("ops").at(0).at("outputs").at(0));
}
#endif

TEST_CASE(
    "encode_phase GEMM attrs round-trip through reconstruct",
    "[graph][tensor][codec]")
{
    TensorGraph src("codec_gemm");
    TensorRef a = src.data({2, 3});
    TensorRef b = src.data({3, 4});
    TensorRef c = TensorRef::adopt(
        gt::gemm(a, b, 2.0, false, false, 1, 0));
    (void)c;
    auto const blob = gt::encode_phase(src);
    REQUIRE(blob.at("ops").at(0).at("op_name") == "GEMM");
    REQUIRE(blob.at("ops").at(0).at("attrs").at("alpha") == 2.0);
    REQUIRE(blob.at("ops").at(0).at("attrs").at("trans_b") == false);
    REQUIRE(blob.at("ops").at(0).at("attrs").at("ndim") == 1);

    TensorGraph dst("codec_gemm_replay");
    gt::PhaseNodeMap refs;
    gt::apply_phase(dst, blob, refs);
    auto const replay = gt::encode_phase(dst);
    REQUIRE(replay.at("ops").at(0).at("op_name") == "GEMM");
    REQUIRE(replay.at("ops").at(0).at("attrs") ==
        blob.at("ops").at(0).at("attrs"));
}

TEST_CASE(
    "encode_phase ADD COPY_INTERSECTION RANDN ADAM families",
    "[graph][tensor][codec]")
{
    TensorGraph add_g("codec_add");
    TensorRef ax = add_g.data({2, 2});
    TensorRef ay = add_g.data({2, 2});
    TensorRef az = TensorRef::adopt(gt::add(1.5, ax, 0.5, ay));
    (void)az;
    auto const add_blob = gt::encode_phase(add_g);
    REQUIRE(add_blob.at("ops").at(0).at("op_name") == "ADD");
    REQUIRE(add_blob.at("ops").at(0).at("attrs").at("alpha") == 1.5);
    REQUIRE(add_blob.at("ops").at(0).at("attrs").at("beta") == 0.5);

    TensorGraph copy_g("codec_copy_isect");
    TensorRef cs = copy_g.data({2, 2});
    TensorRef cd = copy_g.data({2, 2});
    gt::copy_intersection(cs, {0, 0}, cd, {0, 0});
    auto const copy_blob = gt::encode_phase(copy_g);
    REQUIRE(copy_blob.at("ops").at(0).at("op_name") ==
        "COPY_INTERSECTION");
    REQUIRE(copy_blob.at("ops").at(0).at("attrs").at("src_offset") ==
        nlohmann::json::array({0, 0}));

    TensorGraph rand_g("codec_randn");
    TensorRef rd = rand_g.data({2, 2});
    gt::randn(rd, {0, 0}, {2, 2}, 7ull, 0.0, 1.0);
    auto const rand_blob = gt::encode_phase(rand_g);
    REQUIRE(rand_blob.at("ops").at(0).at("op_name") == "RANDN");
    REQUIRE(rand_blob.at("ops").at(0).at("attrs").at("seed") == 7);

    TensorGraph adam_g("codec_adam");
    TensorRef grad = adam_g.data({2});
    TensorRef m = adam_g.data({2});
    TensorRef v = adam_g.data({2});
    TensorRef p = adam_g.data({2});
    gt::adam_step(3, 0.9, 0.999, 1.0e-8, 1.0e-3, 0.01, grad, m, v, p);
    auto const adam_blob = gt::encode_phase(adam_g);
    REQUIRE(adam_blob.at("ops").at(0).at("op_name") == "ADAM_STEP");
    REQUIRE(adam_blob.at("ops").at(0).at("attrs").at("num_iter") == 3);
    REQUIRE(
        adam_blob.at("ops").at(0).at("attrs").at("weight_decay")
            .get<Scalar>() == Scalar(0.01));

    TensorGraph replay("codec_family_replay");
    gt::PhaseNodeMap refs;
    gt::apply_phase(replay, add_blob, refs);
    auto const add_replay = gt::encode_phase(replay);
    REQUIRE(add_replay.at("ops").at(0).at("attrs") ==
        add_blob.at("ops").at(0).at("attrs"));
}

TEST_CASE(
    "encode_phase CONV2D and EMBEDDING INT64 attrs",
    "[graph][tensor][codec]")
{
    TensorGraph conv_g("codec_conv");
    TensorRef X = conv_g.data({4, 4, 1, 1});
    TensorRef C = conv_g.data({3, 3, 1, 1});
    TensorRef Y = conv_g.data({2, 2, 1, 1});
    gt::conv2d_inplace(
        1.0, X, C, 0.0, Y, {0, 0}, {1, 1}, {1, 1});
    auto const conv_blob = gt::encode_phase(conv_g);
    REQUIRE(conv_blob.at("ops").at(0).at("op_name") == "CONV2D_INPLACE");
    REQUIRE(conv_blob.at("ops").at(0).at("attrs").at("padding") ==
        nlohmann::json::array({0, 0}));
    REQUIRE(conv_blob.at("ops").at(0).at("attrs").at("stride") ==
        nlohmann::json::array({1, 1}));

    TensorGraph emb_g("codec_emb");
    TensorRef index = emb_g.data({5, 4}, DataType::INT64);
    TensorRef vocab = emb_g.data({10, 100});
    TensorRef embed = emb_g.data({5, 4, 100});
    gt::embedding(index, vocab, embed, 2);
    auto const emb_blob = gt::encode_phase(emb_g);
    REQUIRE(emb_blob.at("ops").at(0).at("op_name") == "EMBEDDING");
    REQUIRE(emb_blob.at("ops").at(0).at("attrs").at("axis") == 2);
    bool saw_int64 = false;
    for (auto const &node : emb_blob.at("nodes"))
    {
        if (node.at("dtype") == "int64")
        {
            saw_int64 = true;
        }
    }
    REQUIRE(saw_int64);

    TensorGraph replay("codec_emb_replay");
    gt::PhaseNodeMap refs;
    gt::apply_phase(replay, emb_blob, refs);
    auto const replay_blob = gt::encode_phase(replay);
    REQUIRE(replay_blob.at("ops").at(0).at("op_name") == "EMBEDDING");
    REQUIRE(replay_blob.at("ops").at(0).at("attrs").at("axis") == 2);
}

TEST_CASE("decode_phase GELU JSON is not UnknownOp", "[graph][tensor][codec]")
{
    nlohmann::json blob;
    blob["nodes"] = nlohmann::json::array();
    blob["ops"] = nlohmann::json::array(
        {
            {
                {"op_name", "GELU"},
                {"inputs", nlohmann::json::array({1})},
                {"outputs", nlohmann::json::array({2})},
                {"attrs", nlohmann::json::object()},
            },
        });
    auto const decoded = gt::decode_phase(blob);
    REQUIRE(decoded.second.at(0).at("op_name") == "GELU");
}

