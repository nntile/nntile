/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/compiled/log_scalar.cc
 * Test for compiled graph log_scalar operation.
 *
 * @version 1.1.0
 * */

#include "compiled_test_utils.hh"

#include "nntile/tensor/log_scalar.hh"

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::graph::test;

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph LogScalar",
    "[graph][verification]")
{
    // Test that log_scalar compiles and executes (logs only, no output to verify)
    // log_scalar requires a scalar (0D) tensor
    auto build_graph = [](LogicalGraph& g) {
        auto& x = g.tensor(std::vector<Index>{}, "x", DataType::FP32);
        log_scalar(x, "test_value");
    };

    LogicalGraph g("test");
    build_graph(g);
    auto compiled = CompiledGraph::compile(g);
    compiled.bind_data("x", std::vector<float>{1.5f});
    compiled.execute();
    compiled.wait();

    REQUIRE(g.tensors().size() >= 1);
}
