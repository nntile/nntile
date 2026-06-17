/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/tensor_graph/subtract_indexed_outputs.cc
 * Test TensorGraph subtract_indexed_outputs against
 * nntile::tensor::subtract_indexed_outputs.
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/ops/subtract_indexed_outputs.hh"

#include "context_fixture.hh"
#include "nntile/tensor.hh"
#include "nntile/tensor/axis_descriptor.hh"
#include "nntile/tile.hh"
#include "nntile/tensor/ops/subtract_indexed_outputs.hh"
#include "nntile/tensor.hh"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <cstdint>
#include <numeric>

using namespace nntile;
using namespace nntile;
namespace gt = nntile::tensor;

namespace
{

constexpr Scalar val = 1.0;
constexpr Index ignore_index = -1;
constexpr float tolerance = 1e-5f;
constexpr int distr_rank_single = 0;

} 

TEST_CASE("TensorGraph subtract_indexed_outputs structure", "[graph][tensor]")
{
    TensorGraph graph("test");

    auto *labels = graph.data({4}, DataType::INT64)->set_name("labels");
    auto *dst = graph.data({4, 5})->set_name("dst");

    gt::subtract_indexed_outputs(val, labels, dst, ignore_index);

    REQUIRE(graph.num_data() == 2);
    REQUIRE(graph.num_ops() == 1);

    const auto &ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "SUBTRACT_INDEXED_OUTPUTS");
    REQUIRE(ops[0]->inputs().size() == 2);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == dst);
}

TEST_CASE(
    "TensorGraph subtract_indexed_outputs rejects null", "[graph][tensor]")
{
    TensorGraph graph("test");
    auto *labels = graph.data({4}, DataType::INT64)->set_name("labels");
    auto *dst = graph.data({4, 5})->set_name("dst");

    REQUIRE_THROWS_AS(
        gt::subtract_indexed_outputs(val, nullptr, dst, ignore_index),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        gt::subtract_indexed_outputs(val, labels, nullptr, ignore_index),
        std::invalid_argument);
}

TEST_CASE("TensorGraph subtract_indexed_outputs rejects non-INT64 labels",
    "[graph][tensor]")
{
    TensorGraph graph("test");
    auto *labels = graph.data({4})->set_name("labels"); // FP32 default
    auto *dst = graph.data({4, 5})->set_name("dst");

    REQUIRE_THROWS_AS(
        gt::subtract_indexed_outputs(val, labels, dst, ignore_index),
        std::invalid_argument);
}

TEST_CASE("TensorGraph subtract_indexed_outputs rejects ndim mismatch",
    "[graph][tensor]")
{
    TensorGraph graph("test");
    auto *labels = graph.data({4}, DataType::INT64)->set_name("labels");
    // dst has ndim=3 (labels.ndim+2), but must be labels.ndim+1
    auto *dst = graph.data({3, 4, 5})->set_name("dst");

    REQUIRE_THROWS_AS(
        gt::subtract_indexed_outputs(val, labels, dst, ignore_index),
        std::invalid_argument);
}