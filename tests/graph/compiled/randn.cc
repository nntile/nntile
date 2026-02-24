/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/compiled/randn.cc
 * Test for compiled graph randn operation.
 *
 * @version 1.1.0
 * */

#include "compiled_test_utils.hh"

#include "nntile/tensor/randn.hh"

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::graph::test;

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph Randn",
    "[graph][verification]")
{
    // Test that randn compiles and executes (random output, just verify no crash)
    auto build_graph = [](LogicalGraph& g) {
        auto& x = g.tensor({4, 6}, "x", DataType::FP32);
        randn(x, {0, 0}, {4, 6}, 0, 0.0f, 1.0f);
    };

    LogicalGraph g("test");
    build_graph(g);
    auto compiled = CompiledGraph::compile(g);
    compiled.bind_data("x", make_pattern<float>(24, 0.1f));
    compiled.execute();
    compiled.wait();

    auto output = compiled.get_output<float>("x");
    REQUIRE(output.size() == 24u);
}
