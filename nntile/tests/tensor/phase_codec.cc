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
#include <nntile/tensor/ops/gather.hh>
#include <nntile/tensor/ops/gelu.hh>
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
    constexpr std::size_t n_ops =
        sizeof(gt::kV1PhaseOps) / sizeof(gt::kV1PhaseOps[0]);
    REQUIRE(decoded.second.size() == n_ops);
    REQUIRE(n_ops == 6);
    REQUIRE(decoded.second.at(0).at("op_name") == "TORCH_UNARY");
    REQUIRE(decoded.second.at(1).at("op_name") == "TORCH_BINARY");
    REQUIRE(decoded.second.at(2).at("op_name") == "TORCH_TERNARY");
    REQUIRE(decoded.second.at(3).at("op_name") == "GATHER");
    REQUIRE(decoded.second.at(4).at("op_name") == "SCATTER");
    REQUIRE(decoded.second.at(5).at("op_name") == "UNREGISTER");
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
    "encode_phase TensorGraph unknown op fails closed",
    "[graph][tensor][codec]")
{
    TensorGraph graph("codec_gelu");
    TensorRef x = graph.data({2, 2});
    gt::gelu(x);
    REQUIRE_THROWS_WITH(
        gt::encode_phase(graph),
        Catch::Matchers::ContainsSubstring("UnknownOp"));
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
