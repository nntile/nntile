/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/compiled/clear.cc
 * Test for compiled graph clear operation.
 *
 * @version 1.1.0
 * */

#include "compiled_test_utils.hh"

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::graph::test;

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph ClearBool vs Tensor",
    "[graph][verification]")
{
    LogicalGraph g("test");
    auto& x = g.tensor({4}, "x", DataType::BOOL);
    x.mark_input(true);
    x.mark_output(true);
    clear(x);

    auto compiled = CompiledGraph::compile(g);

    std::vector<char> x_data = {static_cast<char>(true), static_cast<char>(false), static_cast<char>(true), static_cast<char>(false)};
    compiled.bind_data("x", x_data);

    compiled.execute();
    compiled.wait();

    auto out = compiled.get_output<char>("x");
    REQUIRE(out.size() == 4);
    for(const auto& v : out)
    {
        REQUIRE(static_cast<bool>(v) == false);
    }
}
